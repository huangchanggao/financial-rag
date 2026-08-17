from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from section_router import route_section

BASE_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_index"


def load_vectorstore():

    print("正在載入 embedding model...")

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


def search(vectorstore, query, k=5):

    section = route_section(query)
    
    print("\nRouter 回傳：")
    print(repr(section))

    if section:
        print(f"\n預測 Section: {section}")

        results = vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter={
                "section": section
            },
            fetch_k=500
        )

    else:
        print("\n無法判斷 Section，搜尋全部文件")

        results = vectorstore.similarity_search_with_score(
            query,
            k=k
        )

    return results


def main():

    vectorstore = load_vectorstore()
    
    print("\nFAISS 中的 Section：")
    
    sections = set()
    
    for doc in vectorstore.docstore._dict.values():
        value = doc.metadata.get("section")
    
        if value:
            sections.add(value)
    
    for section_name in sorted(sections):
        print(repr(section_name))

    query = input("\n請輸入問題：")

    results = search(
        vectorstore,
        query,
        k=5
    )

    print("\n搜尋結果：")

    for i, (document, score) in enumerate(
        results,
        start=1
    ):

        print("\n" + "=" * 80)
        print(f"Result {i}")
        print(f"Score: {score}")

        print("\nMetadata:")
        print(document.metadata)

        print("\nContent:")
        print(document.page_content)


if __name__ == "__main__":
    main()