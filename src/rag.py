from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


BASE_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_index"

# 讀取 .env
load_dotenv(BASE_DIR / ".env")


def load_vectorstore():
    print("正在載入 Embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("正在載入 FAISS index...")

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def retrieve_documents(vectorstore, question, k=5):
    documents = vectorstore.similarity_search(
        question,
        k=k
    )

    return documents


def build_context(documents):
    context_parts = []

    for document in documents:

        chunk_id = document.metadata.get(
            "chunk_id",
            "unknown"
        )

        ticker = document.metadata.get(
            "ticker",
            "unknown"
        )

        form_type = document.metadata.get(
            "form_type",
            "unknown"
        )

        filing_date = document.metadata.get(
            "filing_date",
            "unknown"
        )
        
        section = document.metadata.get(
            "section",
            "unknown"
        )

        source_label = (
            f"{ticker} {form_type}, "
            f"{filing_date}, "
            f"{section}, "
            f"Chunk {chunk_id}"
        )

        context_parts.append(
            f"""
[Source: {source_label}]

{document.page_content}
"""
        )

    return "\n".join(context_parts)


def generate_answer(question, context):
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    prompt = f"""
You are a financial document question-answering assistant.

Answer the user's question using ONLY the information
contained in the provided SEC filing context.

Rules:
1. Do not use outside knowledge.
2. If the context does not contain enough information,
   say that the available filing context is insufficient.
3. Do not invent financial numbers or facts.
4. Cite the relevant sources using the exact source labels
   provided in the context.
5. Example citation:
   [NVDA 10-K, 2026-02-25, Chunk 212]
6. Give a concise but complete answer.

Question:
{question}

SEC Filing Context:
{context}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content


def main():
    vectorstore = load_vectorstore()

    question = input(
        "\n請輸入你的問題："
    )

    print("\n正在檢索相關財報內容...")

    documents = retrieve_documents(
        vectorstore,
        question,
        k=5
    )

    context = build_context(documents)

    print("正在產生回答...\n")

    answer = generate_answer(
        question,
        context
    )

    print("=" * 80)
    print("回答：")
    print(answer)

    print("\n" + "=" * 80)
    print("Retrieved Sources:")

    for i, document in enumerate(
        documents,
        start=1
    ):
        print(
            f"{document.metadata.get('ticker')} "
            f"{document.metadata.get('form_type')} | "
            f"{document.metadata.get('filing_date')} | "
            f"{document.metadata.get('section')} | "
            f"Chunk {document.metadata.get('chunk_id')}"
        )


if __name__ == "__main__":
    main()