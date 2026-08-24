from langchain_groq import ChatGroq


DEFAULT_MODEL = "openai/gpt-oss-20b"


def get_llm(
    model=DEFAULT_MODEL,
    temperature=0
):
    return ChatGroq(
        model=model,
        temperature=temperature
    )