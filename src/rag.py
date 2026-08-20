from pathlib import Path
from dotenv import load_dotenv
import sqlite3

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

from section_router import route_ticker


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "database" / "financial.db"

VECTORSTORE_DIR = (
    BASE_DIR
    / "vectorstore"
    / "faiss_index"
)

# 讀取 .env
load_dotenv(BASE_DIR / ".env")


# =========================================================
# Load models / indexes
# =========================================================

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


def load_reranker():
    print("正在載入 reranker...")

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return reranker


# =========================================================
# Database metadata
# =========================================================

def get_company_sections(ticker):
    """
    從 SQLite 取得指定公司的所有 10-K section 與 title。
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT DISTINCT
            chunks.section,
            chunks.title
        FROM chunks
        JOIN filings
            ON chunks.filing_id = filings.id
        JOIN companies
            ON filings.company_id = companies.id
        WHERE companies.ticker = ?
        ORDER BY chunks.section
        """,
        (ticker,)
    )

    rows = cursor.fetchall()

    conn.close()

    sections = []

    for section, title in rows:
        sections.append({
            "section": section,
            "title": title or ""
        })

    return sections


# =========================================================
# LLM section selector
# =========================================================

def select_section_with_llm(
    question,
    ticker,
    sections
):
    """
    讓 LLM 從該公司的所有 section 中選出
    最可能回答問題的一個 section。
    """

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    section_text = "\n".join(
        [
            f"{section['section']} - "
            f"{section['title']}"
            for section in sections
        ]
    )

    prompt = f"""
You are selecting the most relevant section
from a company's SEC 10-K filing.

Company:
{ticker}

Question:
{question}

Available sections:
{section_text}

Select exactly ONE section that is most likely
to contain the information needed to answer
the question.

Return ONLY the section identifier.

Examples of valid output:
Item 1
Item 1A
Item 1C
Item 7
Item 7A

Do not explain your answer.
"""

    response = llm.invoke(prompt)

    selected = response.content.strip()

    # 驗證 LLM 回傳的 section 是否真的存在
    valid_sections = {
        section["section"]
        for section in sections
    }

    if selected not in valid_sections:
        raise ValueError(
            f"LLM 回傳無效 section: {selected}"
        )

    return selected


# =========================================================
# Retrieval
# =========================================================

def retrieve_documents(
    vectorstore,
    reranker,
    question,
    k=5,
    candidate_k=15
):
    """
    Current retrieval workflow:

    Question
        ↓
    route_ticker
        ↓
    SQLite 取得公司全部 section
        ↓
    LLM 選 1 個 section
        ↓
    selected section 內用 FAISS 找 candidate chunks
        ↓
    Cross-Encoder rerank
        ↓
    Final Top K
    """

    # -----------------------------------------------------
    # 1. Company routing
    # -----------------------------------------------------

    ticker = route_ticker(question)

    # 現階段先不處理無法辨識 ticker 的完整 fallback
    if not ticker:
        print("無法辨識公司 ticker。")
        return []

    print(
        "\nDetected ticker:",
        ticker
    )

    # -----------------------------------------------------
    # 2. 取得公司的 section metadata
    # -----------------------------------------------------

    sections = get_company_sections(
        ticker
    )

    if not sections:
        print(
            f"找不到 {ticker} 的 section metadata。"
        )
        return []

    # -----------------------------------------------------
    # 3. LLM section selection
    # -----------------------------------------------------

    selected_section = select_section_with_llm(
        question,
        ticker,
        sections
    )

    print(
        "LLM selected section:",
        selected_section
    )

    # -----------------------------------------------------
    # 4. FAISS candidate retrieval
    # -----------------------------------------------------

    candidates = vectorstore.similarity_search(
        question,
        k=candidate_k,
        filter={
            "ticker": ticker,
            "section": selected_section
        },
        fetch_k=2000
    )

    if not candidates:
        print(
            "選定 section 中找不到相關 chunks。"
        )
        return []

    print(
        f"FAISS candidate chunks: {len(candidates)}"
    )

    # -----------------------------------------------------
    # 5. Cross-Encoder reranking
    # -----------------------------------------------------

    pairs = [
        (
            question,
            document.page_content
        )
        for document in candidates
    ]

    scores = reranker.predict(
        pairs
    )

    ranked_results = list(
        zip(
            candidates,
            scores
        )
    )

    # Cross-Encoder:
    # score 越高代表相關性越高
    ranked_results.sort(
        key=lambda result: result[1],
        reverse=True
    )

    print("\nReranked candidates:")

    for document, score in ranked_results:
        print(
            document.metadata.get("section"),
            "| Chunk",
            document.metadata.get("chunk_id"),
            "| score:",
            round(float(score), 4)
        )

    # -----------------------------------------------------
    # 6. Final Top K
    # -----------------------------------------------------

    documents = [
        document
        for document, score
        in ranked_results[:k]
    ]

    return documents


# =========================================================
# Build RAG context
# =========================================================

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

    return "\n".join(
        context_parts
    )


# =========================================================
# Answer generation
# =========================================================

def generate_answer(
    question,
    context
):
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
   [NVDA 10-K, 2026-02-25, Item 1A, Chunk 212]
6. Give a concise but complete answer.

Question:
{question}

SEC Filing Context:
{context}

Answer:
"""

    response = llm.invoke(
        prompt
    )

    return response.content


# =========================================================
# Main
# =========================================================

def main():
    vectorstore = load_vectorstore()
    reranker = load_reranker()

    question = input(
        "\n請輸入你的問題："
    )

    print(
        "\n正在檢索相關財報內容..."
    )

    documents = retrieve_documents(
        vectorstore,
        reranker,
        question,
        k=5,
        candidate_k=15
    )

    # 如果 retrieval 失敗
    if not documents:
        print(
            "\n沒有取得可用的文件內容。"
        )
        return

    context = build_context(
        documents
    )

    print(
        "\n正在產生回答...\n"
    )

    answer = generate_answer(
        question,
        context
    )

    print("=" * 80)
    print("回答：")
    print(answer)

    print(
        "\n" + "=" * 80
    )

    print(
        "Retrieved Sources:"
    )

    for i, document in enumerate(
        documents,
        start=1
    ):
        print(
            f"{i}. "
            f"{document.metadata.get('ticker')} "
            f"{document.metadata.get('form_type')} | "
            f"{document.metadata.get('filing_date')} | "
            f"{document.metadata.get('section')} | "
            f"Chunk {document.metadata.get('chunk_id')}"
        )


if __name__ == "__main__":
    main()