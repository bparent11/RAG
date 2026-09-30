from langchain_ollama import OllamaLLM

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents.base import Document

def generate_first_answer(query: str, relevant_docs: list[Document]) -> str:
    model_input = f"""
    Documents:
        {"\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(relevant_docs)])}
    Query:
        {query}

    Based on the sources, answer the query.
    """

    print(f"MODEL INPUT: {model_input}")

    model = OllamaLLM(model="llama3.2:1b")

    messages = [
        SystemMessage(content="You are a helpful assistant. Here are some documents that you have to use to answer correctly to the query."),
        HumanMessage(content=model_input)
    ]

    print("Generating answer ...")
    result = model.invoke(messages)

    return result