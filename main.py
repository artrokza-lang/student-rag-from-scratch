"""Neuro-employee: RAG over Google Docs, powered by OpenAI + ChromaDB + Gradio.

Run:
    python main.py

Requires .env with OPENAI_API_KEY (see .env.example).
"""

from __future__ import annotations

import os
import re

import gradio as gr
import requests
import tiktoken
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from openai import OpenAI

from config import MODELS

# Load OPENAI_API_KEY and any other secrets from .env into os.environ.
load_dotenv()

# Fail fast with a clear message instead of an obscure OpenAI error.
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Create a .env file next to main.py "
        "with the line: OPENAI_API_KEY=sk-..."
    )


class GPT:
    """Neuro-employee: a thin RAG wrapper around OpenAI chat + ChromaDB."""

    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        self.log: str = ""
        self.model: str = model
        self.search_index: Chroma | None = None
        # Pick up the key from env (loaded via dotenv in module scope).
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # ------------------------------------------------------------------ #
    # Ingestion                                                          #
    # ------------------------------------------------------------------ #

    def load_search_indexes(self, url: str) -> Chroma:
        """Download a Google Doc as plain text and index it in ChromaDB."""
        # Extract the document ID from a Google Docs URL.
        match_ = re.search(r"/document/d/([a-zA-Z0-9-_]+)", url)
        if match_ is None:
            raise ValueError(f"Invalid Google Docs URL: {url}")
        doc_id = match_.group(1)

        # Export endpoint returns the document as plain .txt.
        response = requests.get(
            f"https://docs.google.com/document/d/{doc_id}/export",
            params={"format": "txt"},
            timeout=30,
        )
        response.raise_for_status()

        return self.create_embedding(response.text)

    def num_tokens_from_string(self, string: str) -> int:
        """Return the number of tokens in a string, using the model's tokenizer."""
        encoding = tiktoken.encoding_for_model(self.model)
        return len(encoding.encode(string))

    def create_embedding(self, data: str) -> Chroma:
        """Split text into chunks, embed them, and store them in ChromaDB."""
        source_chunks: list[Document] = []
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1024,
            chunk_overlap=0,
        )
        for chunk in splitter.split_text(data):
            source_chunks.append(Document(page_content=chunk, metadata={}))

        count_token = self.num_tokens_from_string(
            " ".join(x.page_content for x in source_chunks)
        )
        self.log += f"Количество токенов в документе: {count_token}\n"

        # Modern LangChain: pass documents= and embedding= explicitly.
        self.search_index = Chroma.from_documents(
            documents=source_chunks,
            embedding=OpenAIEmbeddings(),
        )
        self.log += "Данные из документа загружены в векторную базу данных\n"
        return self.search_index

    # ------------------------------------------------------------------ #
    # Token accounting                                                   #
    # ------------------------------------------------------------------ #

    def num_tokens_from_messages(
        self, messages: list[dict[str, str]], model: str
    ) -> int:
        """Count chat tokens, accounting for per-message overhead."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")

        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4o",
            "gpt-4o-2024-05-13",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4
            tokens_per_name = -1
        elif "gpt-3.5-turbo" in model:
            self.log += (
                "Внимание: gpt-3.5-turbo может обновиться. "
                "Используйте gpt-3.5-turbo-0613.\n"
            )
            return self.num_tokens_from_messages(
                messages, model="gpt-3.5-turbo-0613"
            )
        elif "gpt-4" in model:
            self.log += (
                "Внимание: gpt-4 может обновиться. "
                "Используйте gpt-4-0613.\n"
            )
            return self.num_tokens_from_messages(messages, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"num_tokens_from_messages() is not implemented for model {model}."
            )

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    # ------------------------------------------------------------------ #
    # Generation                                                         #
    # ------------------------------------------------------------------ #

    def answer_index(
        self, system: str, topic: str, temp: float = 1.0
    ) -> str:
        """Retrieve top-k chunks, then call the LLM with them as system context."""
        if not self.search_index:
            self.log += "Модель необходимо обучить!\n"
            return ""

        # Retrieve top-5 chunks by cosine similarity.
        docs = self.search_index.similarity_search(topic, k=5)
        self.log += (
            "Выбираем документы по степени схожести с вопросом "
            "из векторной базы данных:\n"
        )

        # Concatenate chunks with an explicit prefix per chunk.
        # NOTE: original used '\\n' (literal backslash+n) here — fixed to '\n'.
        message_content = re.sub(
            r"\n{2}",
            " ",
            "\n ".join(
                f"Отрывок документа №{i + 1}:\n{doc.page_content}\n"
                for i, doc in enumerate(docs)
            ),
        )
        self.log += f"{message_content}\n"

        messages = [
            {"role": "system", "content": system + message_content},
            {"role": "user", "content": topic},
        ]

        self.log += (
            "\n\nТокенов использовано на вопрос по версии TikToken: "
            f"{self.num_tokens_from_messages(messages, self.model)}\n"
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
        )

        self.log += "\nСтатистика по токенам от языковой модели:\n"
        self.log += f"Токенов использовано всего (вопрос): {completion.usage.prompt_tokens}\n"
        self.log += f"Токенов использовано всего (вопрос-ответ): {completion.usage.total_tokens}\n"

        return completion.choices[0].message.content or ""


# ---------------------------------------------------------------------- #
# Gradio UI                                                              #
# ---------------------------------------------------------------------- #

gpt = GPT("gpt-3.5-turbo")

with gr.Blocks(title="Neuro-employee") as demo:
    # Dropdown: (label, value) pairs — value is the index into MODELS.
    subject = gr.Dropdown(
        [(m["name"], i) for i, m in enumerate(MODELS)],
        label="Данные",
    )
    name = gr.Label(show_label=False)
    prompt = gr.Textbox(label="Промт", interactive=True)
    link = gr.HTML()
    query = gr.Textbox(label="Запрос к LLM", interactive=True)

    def onchange(dropdown: int) -> list:
        """Populate the fields from the selected employee."""
        return [
            MODELS[dropdown]["name"],
            # Collapse tabs and double spaces for cleaner display.
            re.sub(r"\t+|\s\s+", " ", MODELS[dropdown]["prompt"]),
            MODELS[dropdown]["query"],
            f"<a target='_blank' href='{MODELS[dropdown]['doc']}'>"
            f"Документ для обучения</a>",
        ]

    subject.change(
        onchange,
        inputs=[subject],
        outputs=[name, prompt, query, link],
    )

    with gr.Row():
        train_btn = gr.Button("Обучить модель")
        request_btn = gr.Button("Запрос к модели")

    def train(dropdown: int) -> str:
        try:
            gpt.load_search_indexes(MODELS[dropdown]["doc"])
        except Exception as exc:  # noqa: BLE001 — surface any ingestion error
            gpt.log += f"\nОшибка при обучении: {exc}\n"
        return gpt.log

    def predict(p: str, q: str) -> list[str]:
        result = gpt.answer_index(p, q)
        return [result, gpt.log]

    with gr.Row():
        response = gr.Textbox(label="Ответ LLM")
        log = gr.Textbox(label="Логирование")

    train_btn.click(train, [subject], log)
    request_btn.click(predict, [prompt, query], [response, log])

if __name__ == "__main__":
    # inbrowser=True auto-opens the browser — handy in VS Code.
    demo.launch(inbrowser=True, share=False)