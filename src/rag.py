import json
from pathlib import Path
from dotenv import load_dotenv
import sqlite3

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

from src.section_router import route_ticker
from src.llm_client import get_llm



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

    llm = get_llm()

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

def judge_evidence(
    question,
    documents,
    ticker,
    selected_section,
    available_sections
):
    """
    Judge whether the currently retrieved evidence is sufficient.

    Possible retrieval actions:

    1. answer
       Current evidence is sufficient.

    2. same_section
       Current evidence is insufficient, but the missing evidence
       is likely still located in the current section.

    3. new_section
       Current evidence is insufficient, and another section is
       more likely to contain the missing evidence.

    Returns:
        {
            "sufficient": bool,
            "missing_evidence": str,
            "retrieval_action": str,
            "next_section": str | None
        }
    """

    llm = get_llm()

    # -----------------------------------------------------
    # 1. Build evidence context
    # -----------------------------------------------------

    evidence_parts = []

    for document in documents:

        section = document.metadata.get(
            "section",
            "unknown"
        )

        chunk_id = document.metadata.get(
            "chunk_id",
            "unknown"
        )

        evidence_parts.append(
            f"""
[Section: {section}, Chunk: {chunk_id}]

{document.page_content}
"""
        )

    evidence_context = "\n".join(
        evidence_parts
    )

    # -----------------------------------------------------
    # 2. Normalize available sections
    # -----------------------------------------------------

    section_names = []
    section_lines = []

    for section in available_sections:

        if isinstance(section, str):

            section_name = section
            section_title = ""

        elif isinstance(section, dict):

            section_name = section.get(
                "section"
            )

            section_title = section.get(
                "title",
                ""
            )

        else:
            continue

        if not section_name:
            continue

        section_names.append(
            section_name
        )

        if section_title:

            section_lines.append(
                f"{section_name} - {section_title}"
            )

        else:

            section_lines.append(
                section_name
            )

    section_list = "\n".join(
        section_lines
    )

    # -----------------------------------------------------
    # 3. Evidence Judge prompt
    # -----------------------------------------------------

    prompt = f"""
You are an evidence sufficiency judge for a financial RAG system.

Your task is NOT to answer the user's question.

Evaluate whether the currently retrieved SEC filing evidence is
sufficient to answer the question accurately and usefully.

Company:
{ticker}

Question:
{question}

Current section:
{selected_section}

Retrieved evidence:
{evidence_context}

Available sections:
{section_list}

You must choose exactly ONE retrieval action:

"answer"
Use this when the current evidence is sufficient to answer the
question.

"same_section"
Use this when the current evidence is insufficient, but the missing
information is most likely still located elsewhere inside the CURRENT
section.

"new_section"
Use this when the current evidence is insufficient and the missing
information is more likely located in a DIFFERENT section.

Rules:

1. Set "sufficient" to true only when the retrieved evidence directly
   contains enough information for a useful and well-supported answer.

2. Do not require every possible detail. Judge sufficiency based on
   the user's actual question.

3. If the evidence is sufficient:
   - "sufficient" must be true
   - "retrieval_action" must be "answer"
   - "missing_evidence" must be ""
   - "next_section" must be null

4. If important evidence is missing but is likely still in the
   current section:
   - "sufficient" must be false
   - "retrieval_action" must be "same_section"
   - "next_section" must equal the current section

5. If important evidence is likely located in another section:
   - "sufficient" must be false
   - "retrieval_action" must be "new_section"
   - "next_section" must be exactly one section from the available
     section list
   - "next_section" must not equal the current section

6. Briefly describe the important missing information in
   "missing_evidence".

7. Do not invent section identifiers.

8. Make the decision concisely. Do not perform a long analysis.

9. Return ONLY one valid JSON object.
   Do not include markdown.
   Do not include code fences.
   Do not include explanations outside the JSON.

Valid output examples:

{{
    "sufficient": true,
    "missing_evidence": "",
    "retrieval_action": "answer",
    "next_section": null
}}

{{
    "sufficient": false,
    "missing_evidence": "Additional risk factors from the same Risk Factors section.",
    "retrieval_action": "same_section",
    "next_section": "Item 1A"
}}

{{
    "sufficient": false,
    "missing_evidence": "Specific cybersecurity risk details.",
    "retrieval_action": "new_section",
    "next_section": "Item 1A"
}}
"""

    # -----------------------------------------------------
    # 4. Call LLM
    # -----------------------------------------------------

    response = llm.invoke(
        prompt
    )

    raw_output = response.content.strip()

    # -----------------------------------------------------
    # 5. Debug empty output
    # -----------------------------------------------------

    if not raw_output:

        print(
            "\nEvidence Judge returned empty content."
        )

        print(
            "finish_reason:",
            response.response_metadata.get(
                "finish_reason"
            )
        )

        print(
            "token_usage:",
            response.response_metadata.get(
                "token_usage"
            )
        )

        # Fail safely:
        # search more evidence from the same section
        # instead of incorrectly jumping to another section.
        return {
            "sufficient": False,
            "missing_evidence": (
                "Evidence Judge returned no structured result."
            ),
            "retrieval_action": "same_section",
            "next_section": selected_section
        }

    # -----------------------------------------------------
    # 6. Remove accidental Markdown code fences
    # -----------------------------------------------------

    if raw_output.startswith("```"):

        raw_output = raw_output.strip("`")

        if raw_output.startswith("json"):
            raw_output = raw_output[4:].strip()

    # -----------------------------------------------------
    # 7. Parse JSON
    # -----------------------------------------------------

    try:

        result = json.loads(
            raw_output
        )

    except json.JSONDecodeError as error:

        print(
            "\nEvidence Judge JSON parse failed:"
        )

        print(
            "JSON error:",
            error
        )

        print(
            "raw_output:",
            repr(raw_output)
        )

        # Fail safely:
        # search current section again rather than
        # incorrectly choosing another section.
        return {
            "sufficient": False,
            "missing_evidence": (
                "Evidence Judge returned invalid JSON."
            ),
            "retrieval_action": "same_section",
            "next_section": selected_section
        }

    # -----------------------------------------------------
    # 8. Read result
    # -----------------------------------------------------

    sufficient = result.get(
        "sufficient"
    )

    missing_evidence = result.get(
        "missing_evidence",
        ""
    )

    retrieval_action = result.get(
        "retrieval_action"
    )

    next_section = result.get(
        "next_section"
    )

    # -----------------------------------------------------
    # 9. Validate sufficient
    # -----------------------------------------------------

    if not isinstance(
        sufficient,
        bool
    ):

        print(
            "Evidence Judge returned invalid "
            "'sufficient' value."
        )

        return {
            "sufficient": False,
            "missing_evidence": (
                "Evidence Judge returned an invalid decision."
            ),
            "retrieval_action": "same_section",
            "next_section": selected_section
        }

    # -----------------------------------------------------
    # 10. Evidence is sufficient
    # -----------------------------------------------------

    if sufficient:

        return {
            "sufficient": True,
            "missing_evidence": "",
            "retrieval_action": "answer",
            "next_section": None
        }

    # -----------------------------------------------------
    # 11. Validate retrieval action
    # -----------------------------------------------------

    valid_actions = {
        "same_section",
        "new_section"
    }

    if retrieval_action not in valid_actions:

        print(
            "Evidence Judge returned invalid "
            "retrieval_action:",
            retrieval_action
        )

        return {
            "sufficient": False,
            "missing_evidence": missing_evidence,
            "retrieval_action": "same_section",
            "next_section": selected_section
        }

    # -----------------------------------------------------
    # 12. Same-section retrieval
    # -----------------------------------------------------

    if retrieval_action == "same_section":

        return {
            "sufficient": False,
            "missing_evidence": missing_evidence,
            "retrieval_action": "same_section",
            "next_section": selected_section
        }

    # -----------------------------------------------------
    # 13. New-section retrieval
    # -----------------------------------------------------

    if next_section not in section_names:

        print(
            "Evidence Judge selected invalid section:",
            next_section
        )

        return {
            "sufficient": False,
            "missing_evidence": missing_evidence,
            "retrieval_action": "same_section",
            "next_section": selected_section
        }

    if next_section == selected_section:

        print(
            "Evidence Judge requested new_section "
            "but selected the current section."
        )

        return {
            "sufficient": False,
            "missing_evidence": missing_evidence,
            "retrieval_action": "same_section",
            "next_section": selected_section
        }

    # -----------------------------------------------------
    # 14. Return normalized new-section result
    # -----------------------------------------------------

    return {
        "sufficient": False,
        "missing_evidence": missing_evidence,
        "retrieval_action": "new_section",
        "next_section": next_section
    }
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
        return {
            "ticker": None,
            "selected_section": None,
            "documents": []
        }

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
    
        return {
            "ticker": ticker,
            "selected_section": None,
            "documents": []
        }

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
    
        return {
            "ticker": ticker,
            "selected_section": selected_section,
            "documents": []
        }

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

    return {
        "ticker": ticker,
        "selected_section": selected_section,
        "documents": documents
    }

def retrieve_from_section(
    vectorstore,
    reranker,
    question,
    ticker,
    section,
    k=5,
    candidate_k=15
):
    """
    Retrieve and rerank chunks from a specific section.

    Used for the second retrieval round when the
    Evidence Judge determines that the first-round
    evidence is insufficient.
    """

    print(
        "\nSecond-round section:",
        section
    )

    # -----------------------------------------------------
    # 1. FAISS retrieval within selected section
    # -----------------------------------------------------

    candidates = vectorstore.similarity_search(
        question,
        k=candidate_k,
        filter={
            "ticker": ticker,
            "section": section
        },
        fetch_k=2000
    )

    if not candidates:
        print(
            "Second-round section contains no relevant chunks."
        )
        return []

    print(
        f"Second-round FAISS candidates: {len(candidates)}"
    )

    # -----------------------------------------------------
    # 2. Cross-Encoder reranking
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

    ranked_results.sort(
        key=lambda result: result[1],
        reverse=True
    )

    print(
        "\nSecond-round reranked candidates:"
    )

    for document, score in ranked_results:
        print(
            document.metadata.get("section"),
            "| Chunk",
            document.metadata.get("chunk_id"),
            "| score:",
            round(float(score), 4)
        )

    # -----------------------------------------------------
    # 3. Return Top K
    # -----------------------------------------------------

    documents = [
        document
        for document, score
        in ranked_results[:k]
    ]

    return documents

def rerank_documents(
    reranker,
    question,
    documents,
    k=6
):
    """
    Rerank a combined document set and return the final Top K.
    """

    if not documents:
        return []

    pairs = [
        (
            question,
            document.page_content
        )
        for document in documents
    ]

    scores = reranker.predict(
        pairs
    )

    ranked_results = list(
        zip(
            documents,
            scores
        )
    )

    ranked_results.sort(
        key=lambda result: result[1],
        reverse=True
    )

    print(
        "\nFinal combined reranking:"
    )

    for document, score in ranked_results:
        print(
            document.metadata.get("section"),
            "| Chunk",
            document.metadata.get("chunk_id"),
            "| score:",
            round(float(score), 4)
        )

    final_documents = [
        document
        for document, score
        in ranked_results[:k]
    ]

    return final_documents

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
    llm = get_llm()
    
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

def answer_question(
    question,
    vectorstore,
    reranker
):
    """
    Main RAG entry point for API / UI.

    Workflow:

    1. First-round retrieval
    2. Evidence Judge
    3. If needed:
       - same_section retrieval
       - or new_section retrieval
    4. Combined reranking
    5. Answer generation
    """

    # -----------------------------------------------------
    # 1. First-round retrieval
    # -----------------------------------------------------

    retrieval_result = retrieve_documents(
        vectorstore,
        reranker,
        question,
        k=5,
        candidate_k=15
    )

    ticker = retrieval_result[
        "ticker"
    ]

    selected_section = retrieval_result[
        "selected_section"
    ]

    documents = retrieval_result[
        "documents"
    ]

    # -----------------------------------------------------
    # 2. Handle retrieval failure
    # -----------------------------------------------------

    if not documents:

        return {
            "question": question,
            "ticker": ticker,
            "selected_section": selected_section,
            "retrieval_rounds": 1,
            "additional_section": None,
            "answer": (
                "No relevant filing evidence was found."
            ),
            "sources": []
        }

    # -----------------------------------------------------
    # 3. Evidence Judge
    # -----------------------------------------------------

    available_sections = get_company_sections(
        ticker
    )

    judge_result = judge_evidence(
        question=question,
        documents=documents,
        ticker=ticker,
        selected_section=selected_section,
        available_sections=available_sections
    )

    print(
        "\nEvidence Judge:",
        judge_result
    )

    retrieval_rounds = 1
    additional_section = None

    # -----------------------------------------------------
    # 4. Read Judge action
    # -----------------------------------------------------

    retrieval_action = judge_result[
        "retrieval_action"
    ]

    missing_evidence = judge_result[
        "missing_evidence"
    ]

    # -----------------------------------------------------
    # 5. Evidence sufficient
    # -----------------------------------------------------

    if retrieval_action == "answer":

        print(
            "\nEvidence sufficient. "
            "No second retrieval needed."
        )

    # -----------------------------------------------------
    # 6. Same-section second retrieval
    # -----------------------------------------------------

    elif retrieval_action == "same_section":

        retrieval_rounds = 2

        print(
            "\nEvidence insufficient."
        )

        print(
            "Retrieval action: same_section"
        )

        print(
            "Missing evidence:",
            missing_evidence
        )

        print(
            "Searching current section again:",
            selected_section
        )

        # Use the missing evidence to make the second
        # retrieval query more targeted.
        second_query = (
            f"{question}\n"
            f"Missing evidence: {missing_evidence}"
        )

        second_documents = retrieve_from_section(
            vectorstore=vectorstore,
            reranker=reranker,
            question=second_query,
            ticker=ticker,
            section=selected_section,
            k=10,
            candidate_k=30
        )

        # Combine first and second round
        combined_documents = (
            documents
            + second_documents
        )

        # Remove duplicate chunks
        unique_documents = []
        seen_chunk_ids = set()

        for document in combined_documents:

            chunk_id = document.metadata.get(
                "chunk_id"
            )

            if chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(
                chunk_id
            )

            unique_documents.append(
                document
            )

        # Final reranking uses the original user question
        documents = rerank_documents(
            reranker=reranker,
            question=question,
            documents=unique_documents,
            k=6
        )

    # -----------------------------------------------------
    # 7. New-section second retrieval
    # -----------------------------------------------------

    elif retrieval_action == "new_section":

        second_section = judge_result[
            "next_section"
        ]

        retrieval_rounds = 2
        additional_section = second_section

        print(
            "\nEvidence insufficient."
        )

        print(
            "Retrieval action: new_section"
        )

        print(
            "Missing evidence:",
            missing_evidence
        )

        print(
            "Starting second retrieval round:",
            second_section
        )

        # Make second-round retrieval more targeted
        second_query = (
            f"{question}\n"
            f"Missing evidence: {missing_evidence}"
        )

        second_documents = retrieve_from_section(
            vectorstore=vectorstore,
            reranker=reranker,
            question=second_query,
            ticker=ticker,
            section=second_section,
            k=5,
            candidate_k=15
        )

        combined_documents = (
            documents
            + second_documents
        )

        # Remove duplicate chunks
        unique_documents = []
        seen_chunk_ids = set()

        for document in combined_documents:

            chunk_id = document.metadata.get(
                "chunk_id"
            )

            if chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(
                chunk_id
            )

            unique_documents.append(
                document
            )

        # Final combined reranking
        documents = rerank_documents(
            reranker=reranker,
            question=question,
            documents=unique_documents,
            k=6
        )

    # -----------------------------------------------------
    # 8. Build final context
    # -----------------------------------------------------

    context = build_context(
        documents
    )

    # -----------------------------------------------------
    # 9. Generate answer
    # -----------------------------------------------------

    answer = generate_answer(
        question,
        context
    )

    # -----------------------------------------------------
    # 10. Build source metadata
    # -----------------------------------------------------

    sources = []

    for document in documents:

        sources.append({
            "ticker": document.metadata.get(
                "ticker"
            ),
            "form_type": document.metadata.get(
                "form_type"
            ),
            "filing_date": document.metadata.get(
                "filing_date"
            ),
            "section": document.metadata.get(
                "section"
            ),
            "chunk_id": document.metadata.get(
                "chunk_id"
            )
        })

    # -----------------------------------------------------
    # 11. API result
    # -----------------------------------------------------

    return {
        "question": question,
        "ticker": ticker,
        "selected_section": selected_section,
        "retrieval_rounds": retrieval_rounds,
        "additional_section": additional_section,
        "answer": answer,
        "sources": sources
    }

def main():
    vectorstore = load_vectorstore()
    reranker = load_reranker()

    question = input(
        "\n請輸入你的問題："
    )

    print(
        "\n正在檢索相關財報內容..."
    )

    result = answer_question(
        question,
        vectorstore,
        reranker
    )

    print("\n" + "=" * 80)
    print("回答：")
    print(result["answer"])

    print(
        "\n" + "=" * 80
    )

    print("Retrieved Sources:")

    for i, source in enumerate(
        result["sources"],
        start=1
    ):
        print(
            f"{i}. "
            f"{source['ticker']} "
            f"{source['form_type']} | "
            f"{source['filing_date']} | "
            f"{source['section']} | "
            f"Chunk {source['chunk_id']}"
        )


if __name__ == "__main__":
    main()