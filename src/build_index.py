import sqlite3
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "database" / "financial.db"

VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_index"


def load_chunks_from_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        chunks.id,
        chunks.chunk_text,
        chunks.chunk_index,
        chunks.section,
        companies.ticker,
        companies.company_name,
        filings.form_type,
        filings.filing_date,
        filings.source_file
    FROM chunks
    JOIN filings
        ON chunks.filing_id = filings.id
    JOIN companies
        ON filings.company_id = companies.id
    ORDER BY chunks.id
    """)

    rows = cursor.fetchall()

    conn.close()

    documents = []

    for row in rows:
        (
            chunk_id,
            chunk_text,
            chunk_index,
            section,
            ticker,
            company_name,
            form_type,
            filing_date,
            source_file
        ) = row

        document = Document(
            page_content=chunk_text,
            metadata={
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "section": section,
                "ticker": ticker,
                "company": company_name,
                "form_type": form_type,
                "filing_date": filing_date,
                "source_file": source_file
            }
        )

        documents.append(document)

    return documents


def build_faiss_index(documents):

    print("正在載入 embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("正在建立 embeddings + FAISS index...")

    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    VECTORSTORE_DIR.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    vectorstore.save_local(
        str(VECTORSTORE_DIR)
    )

    return vectorstore


def main():

    print("從 SQLite 載入 chunks...")

    documents = load_chunks_from_database()

    print(
        f"共載入 {len(documents)} 個 documents"
    )

    if not documents:
        print("資料庫中沒有 chunks")
        return

    build_faiss_index(documents)

    print("\nFAISS index 建立完成")
    print("儲存位置：")
    print(VECTORSTORE_DIR)


if __name__ == "__main__":
    main()