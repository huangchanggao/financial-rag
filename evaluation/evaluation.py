from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.section_router import route_section


import json
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "evaluation"
    / "holdout_dataset.json"
)
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_index"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    return dataset


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def evaluate_question(vectorstore, item, k=5):
    question = item["question"]
    expected_section = item["expected_section"]
    expected_keywords = item["expected_keywords"]

    # ==================================================
    # 1. Baseline retrieval
    # ==================================================

    baseline_documents = vectorstore.similarity_search(
        question,
        k=k
    )

    baseline_sections = [
        doc.metadata.get("section", "")
        for doc in baseline_documents
    ]

    baseline_hit_at_1 = (
        expected_section in baseline_sections[:1]
    )

    baseline_hit_at_3 = (
        expected_section in baseline_sections[:3]
    )

    baseline_hit_at_5 = (
        expected_section in baseline_sections[:5]
    )

    # ==================================================
    # 2. Section-aware retrieval
    # ==================================================

    predicted_section = route_section(question)

    if predicted_section:
        section_documents = vectorstore.similarity_search(
            question,
            k=k,
            filter={
                "section": predicted_section
            },
            fetch_k=500
        )

    else:
        section_documents = vectorstore.similarity_search(
            question,
            k=k
        )

    section_sections = [
        doc.metadata.get("section", "")
        for doc in section_documents
    ]

    section_hit_at_1 = (
        expected_section in section_sections[:1]
    )

    section_hit_at_3 = (
        expected_section in section_sections[:3]
    )

    section_hit_at_5 = (
        expected_section in section_sections[:5]
    )

    # ==================================================
    # 3. Keyword recall
    # ==================================================

    combined_text = " ".join(
        doc.page_content.lower()
        for doc in section_documents
    )

    keyword_results = {}

    for keyword in expected_keywords:
        keyword_results[keyword] = (
            keyword.lower() in combined_text
        )

    keyword_hits = sum(
        keyword_results.values()
    )

    keyword_recall = (
        keyword_hits / len(expected_keywords)
        if expected_keywords
        else 0
    )

    # ==================================================
    # 4. Return results
    # ==================================================

    return {
        "question": question,

        "expected_section": expected_section,
        "predicted_section": predicted_section,

        "baseline_hit_at_1": baseline_hit_at_1,
        "baseline_hit_at_3": baseline_hit_at_3,
        "baseline_hit_at_5": baseline_hit_at_5,
        "baseline_sections": baseline_sections,

        "section_hit_at_1": section_hit_at_1,
        "section_hit_at_3": section_hit_at_3,
        "section_hit_at_5": section_hit_at_5,
        "section_sections": section_sections,

        "keyword_recall": keyword_recall,
        "keyword_results": keyword_results
    }


def main():
    print("載入 evaluation dataset...")

    dataset = load_dataset()

    print(f"共 {len(dataset)} 題")

    print("\n載入 FAISS index...")

    vectorstore = load_vectorstore()

    results = []

    for i, item in enumerate(dataset, start=1):
        print("\n" + "=" * 80)
        print(f"Question {i}")
        print(item["question"])

        result = evaluate_question(
            vectorstore,
            item,
            k=5
        )

        results.append(result)

        print("\nExpected Section:")
        print(result["expected_section"])

        print("\nPredicted Section:")
        print(result["predicted_section"])

        # ==================================================
        # Baseline
        # ==================================================

        print("\nBaseline Retrieved Sections:")
        for section in result["baseline_sections"]:
            print("-", section)

        print("\nBaseline Hit@1:")
        print(result["baseline_hit_at_1"])

        print("Baseline Hit@3:")
        print(result["baseline_hit_at_3"])

        print("Baseline Hit@5:")
        print(result["baseline_hit_at_5"])

        # ==================================================
        # Section-aware
        # ==================================================

        print("\nSection-aware Retrieved Sections:")
        for section in result["section_sections"]:
            print("-", section)

        print("\nSection-aware Hit@1:")
        print(result["section_hit_at_1"])

        print("Section-aware Hit@3:")
        print(result["section_hit_at_3"])

        print("Section-aware Hit@5:")
        print(result["section_hit_at_5"])

        # ==================================================
        # Keyword
        # ==================================================

        print("\nKeyword Results:")

        for keyword, hit in result["keyword_results"].items():
            print(f"- {keyword}: {hit}")

        print(
            "\nKeyword Recall:",
            round(result["keyword_recall"], 2)
        )

    # ==================================================
    # Summary
    # ==================================================

    total_questions = len(results)

    # Baseline
    baseline_hits_1 = sum(
        1 for result in results
        if result["baseline_hit_at_1"]
    )

    baseline_hits_3 = sum(
        1 for result in results
        if result["baseline_hit_at_3"]
    )

    baseline_hits_5 = sum(
        1 for result in results
        if result["baseline_hit_at_5"]
    )

    baseline_hit_rate_1 = (
        baseline_hits_1 / total_questions
        if total_questions
        else 0
    )

    baseline_hit_rate_3 = (
        baseline_hits_3 / total_questions
        if total_questions
        else 0
    )

    baseline_hit_rate_5 = (
        baseline_hits_5 / total_questions
        if total_questions
        else 0
    )

    # Section-aware
    section_hits_1 = sum(
        1 for result in results
        if result["section_hit_at_1"]
    )

    section_hits_3 = sum(
        1 for result in results
        if result["section_hit_at_3"]
    )

    section_hits_5 = sum(
        1 for result in results
        if result["section_hit_at_5"]
    )

    section_hit_rate_1 = (
        section_hits_1 / total_questions
        if total_questions
        else 0
    )

    section_hit_rate_3 = (
        section_hits_3 / total_questions
        if total_questions
        else 0
    )

    section_hit_rate_5 = (
        section_hits_5 / total_questions
        if total_questions
        else 0
    )

    # Keyword recall
    average_keyword_recall = (
        sum(
            result["keyword_recall"]
            for result in results
        )
        / total_questions
        if total_questions
        else 0
    )

    # ==================================================
    # Print Summary
    # ==================================================

    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)

    print("\nBaseline Retriever")

    print(
        f"Hit@1: "
        f"{baseline_hits_1}/{total_questions} "
        f"({baseline_hit_rate_1:.2%})"
    )

    print(
        f"Hit@3: "
        f"{baseline_hits_3}/{total_questions} "
        f"({baseline_hit_rate_3:.2%})"
    )

    print(
        f"Hit@5: "
        f"{baseline_hits_5}/{total_questions} "
        f"({baseline_hit_rate_5:.2%})"
    )

    print("\nSection-aware Retriever")

    print(
        f"Hit@1: "
        f"{section_hits_1}/{total_questions} "
        f"({section_hit_rate_1:.2%})"
    )

    print(
        f"Hit@3: "
        f"{section_hits_3}/{total_questions} "
        f"({section_hit_rate_3:.2%})"
    )

    print(
        f"Hit@5: "
        f"{section_hits_5}/{total_questions} "
        f"({section_hit_rate_5:.2%})"
    )

    print(
        f"\nAverage Keyword Recall@5: "
        f"{average_keyword_recall:.2%}"
    )


if __name__ == "__main__":
    main()