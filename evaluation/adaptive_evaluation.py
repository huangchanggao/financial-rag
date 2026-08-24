from pathlib import Path
import sys
import json

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.rag import (
    load_vectorstore,
    load_reranker,
    get_company_sections,
    select_section_with_llm,
    judge_evidence,
    retrieve_from_section
)


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "evaluation"
    / "adaptive_retrieval_eval.json"
)


# =========================================================
# Dataset
# =========================================================

def load_dataset():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# =========================================================
# First-round retrieval
# =========================================================

def retrieve_first_round(
    vectorstore,
    reranker,
    question,
    ticker,
    section,
    k=5,
    candidate_k=15
):

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
        return []

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

    ranked = list(
        zip(
            candidates,
            scores
        )
    )

    ranked.sort(
        key=lambda result: result[1],
        reverse=True
    )

    return [
        document
        for document, score
        in ranked[:k]
    ]


# =========================================================
# Section Recall
# =========================================================

def calculate_section_recall(
    retrieved_sections,
    expected_sections
):

    expected_set = set(
        expected_sections
    )

    retrieved_set = set(
        retrieved_sections
    )

    if not expected_set:
        return 0.0

    hits = len(
        expected_set
        & retrieved_set
    )

    return (
        hits
        / len(expected_set)
    )


# =========================================================
# Evaluate one question
# =========================================================

def evaluate_question(
    vectorstore,
    reranker,
    item
):

    question = item["question"]
    ticker = item["ticker"]

    expected_sections = item[
        "expected_sections"
    ]

    question_type = item[
        "question_type"
    ]

    # -----------------------------------------------------
    # Get available sections
    # -----------------------------------------------------

    available_sections = get_company_sections(
        ticker
    )

    # -----------------------------------------------------
    # Baseline:
    # exactly one selected section
    # -----------------------------------------------------

    first_section = select_section_with_llm(
        question,
        ticker,
        available_sections
    )

    first_documents = retrieve_first_round(
        vectorstore=vectorstore,
        reranker=reranker,
        question=question,
        ticker=ticker,
        section=first_section,
        k=5,
        candidate_k=15
    )

    baseline_sections = [
        first_section
    ]

    baseline_recall = calculate_section_recall(
        baseline_sections,
        expected_sections
    )

    # -----------------------------------------------------
    # Adaptive retrieval
    # -----------------------------------------------------

    judge_result = judge_evidence(
        question=question,
        documents=first_documents,
        ticker=ticker,
        selected_section=first_section,
        available_sections=available_sections
    )

    retrieval_action = judge_result[
        "retrieval_action"
    ]

    missing_evidence = judge_result[
        "missing_evidence"
    ]

    adaptive_sections = [
        first_section
    ]

    retrieval_rounds = 1
    second_section = None

    # -----------------------------------------------------
    # Action 1: answer
    # -----------------------------------------------------

    if retrieval_action == "answer":

        pass

    # -----------------------------------------------------
    # Action 2: same section
    # -----------------------------------------------------

    elif retrieval_action == "same_section":

        retrieval_rounds = 2
        second_section = first_section

        second_query = (
            f"{question}\n"
            f"Missing evidence: {missing_evidence}"
        )

        # Important:
        # same section is NOT appended again to
        # adaptive_sections because section coverage
        # has not increased.
        retrieve_from_section(
            vectorstore=vectorstore,
            reranker=reranker,
            question=second_query,
            ticker=ticker,
            section=first_section,
            k=10,
            candidate_k=30
        )

    # -----------------------------------------------------
    # Action 3: new section
    # -----------------------------------------------------

    elif retrieval_action == "new_section":

        second_section = judge_result[
            "next_section"
        ]

        retrieval_rounds = 2

        if (
            second_section
            and second_section not in adaptive_sections
        ):
            adaptive_sections.append(
                second_section
            )

        second_query = (
            f"{question}\n"
            f"Missing evidence: {missing_evidence}"
        )

        retrieve_from_section(
            vectorstore=vectorstore,
            reranker=reranker,
            question=second_query,
            ticker=ticker,
            section=second_section,
            k=5,
            candidate_k=15
        )

    # -----------------------------------------------------
    # Section recall
    # -----------------------------------------------------

    adaptive_recall = calculate_section_recall(
        adaptive_sections,
        expected_sections
    )

    return {
        "id": item["id"],
        "question": question,
        "ticker": ticker,
        "question_type": question_type,

        "expected_sections":
            expected_sections,

        "baseline_sections":
            baseline_sections,

        "baseline_recall":
            baseline_recall,

        "adaptive_sections":
            adaptive_sections,

        "adaptive_recall":
            adaptive_recall,

        "retrieval_rounds":
            retrieval_rounds,

        "retrieval_action":
            retrieval_action,

        "second_section":
            second_section
    }


# =========================================================
# Dataset evaluation
# =========================================================

def evaluate_dataset(
    vectorstore,
    reranker,
    dataset
):

    results = []

    for item in dataset:

        print(
            "\n" + "=" * 80
        )

        print(
            f"Question {item['id']}: "
            f"{item['question']}"
        )

        result = evaluate_question(
            vectorstore,
            reranker,
            item
        )

        results.append(
            result
        )

        print(
            "Expected:",
            result["expected_sections"]
        )

        print(
            "Baseline:",
            result["baseline_sections"],
            "| Recall:",
            result["baseline_recall"]
        )

        print(
            "Adaptive:",
            result["adaptive_sections"],
            "| Recall:",
            result["adaptive_recall"]
        )

        print(
            "Retrieval rounds:",
            result["retrieval_rounds"]
        )
        
        print(
            "Retrieval action:",
            result["retrieval_action"]
        )
        
        print(
            "Second section:",
            result["second_section"]
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    total = len(results)

    baseline_average = sum(
        result["baseline_recall"]
        for result in results
    ) / total

    adaptive_average = sum(
        result["adaptive_recall"]
        for result in results
    ) / total

    average_rounds = sum(
        result["retrieval_rounds"]
        for result in results
    ) / total

    second_round_count = sum(
        1
        for result in results
        if result["retrieval_rounds"] == 2
    )
    
    same_section_count = sum(
        1
        for result in results
        if result["retrieval_action"]
        == "same_section"
    )
    
    new_section_count = sum(
        1
        for result in results
        if result["retrieval_action"]
        == "new_section"
    )
    
    answer_count = sum(
        1
        for result in results
        if result["retrieval_action"]
        == "answer"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        f"\nBaseline Expected Section Recall: "
        f"{baseline_average:.2%}"
    )

    print(
        f"Adaptive Expected Section Recall: "
        f"{adaptive_average:.2%}"
    )

    print(
        f"Average Retrieval Rounds: "
        f"{average_rounds:.2f}"
    )

    print(
        f"Second-round Trigger Rate: "
        f"{second_round_count}/{total} "
        f"({second_round_count / total:.2%})"
    )
    
    print(
        f"Direct Answer: "
        f"{answer_count}/{total} "
        f"({answer_count / total:.2%})"
    )
    
    print(
        f"Same-section Retry: "
        f"{same_section_count}/{total} "
        f"({same_section_count / total:.2%})"
    )
    
    print(
        f"New-section Retrieval: "
        f"{new_section_count}/{total} "
        f"({new_section_count / total:.2%})"
    )

    # -----------------------------------------------------
    # Breakdown by question type
    # -----------------------------------------------------

    for question_type in [
        "single_section",
        "cross_section"
    ]:

        subset = [
            result
            for result in results
            if result["question_type"]
            == question_type
        ]

        if not subset:
            continue

        baseline = sum(
            result["baseline_recall"]
            for result in subset
        ) / len(subset)

        adaptive = sum(
            result["adaptive_recall"]
            for result in subset
        ) / len(subset)

        print(
            f"\n{question_type}"
        )

        print(
            f"Baseline Recall: "
            f"{baseline:.2%}"
        )

        print(
            f"Adaptive Recall: "
            f"{adaptive:.2%}"
        )


# =========================================================
# Main
# =========================================================

def main():

    print(
        "Loading adaptive retrieval dataset..."
    )

    dataset = load_dataset()

    print(
        f"Questions: {len(dataset)}"
    )

    print(
        "\nLoading vectorstore..."
    )

    vectorstore = load_vectorstore()

    print(
        "\nLoading reranker..."
    )

    reranker = load_reranker()

    evaluate_dataset(
        vectorstore,
        reranker,
        dataset
    )


if __name__ == "__main__":
    main()