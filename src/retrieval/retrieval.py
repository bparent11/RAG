import os

from langchain_core.documents.base import Document

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

persistent_dir = os.environ.get("PERSISTENT_DIRECTORY", "db_chroma_db")

embedding_model = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory=persistent_dir,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

def retrieve_k_similar_docs(query: str, k: int = 5) -> list[Document] | str:
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k":5,
            "score_threshold":0.6
        }
    )

    relevant_docs = retriever.invoke(query)

    if relevant_docs:
        print(f"{len(relevant_docs)} docs found!")
        return relevant_docs
    
    empty_doc = Document
    empty_doc.page_content = "Any reliable source found."

    return empty_doc

def generate_prompt(query: str, relevant_docs: list[Document]) -> str:
    model_input = f"""
    Sources:
        {"\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(relevant_docs)])}
    Query:
        {query}

    Based on the sources, answer the query.
    """

    print(f"MODEL INPUT: {model_input}")

    model = OllamaLLM(model="llama3.2:1b")

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=model_input)
    ]

    print("Generating answer ...")
    result = model.invoke(messages)

    return result
    
def generation_pipeline(query:str):
    relevants_docs = retrieve_k_similar_docs(
        query=query,
        k=5
    )
    answer = generate_prompt(
        query=query,
        relevant_docs=relevants_docs
    )

    return answer


if __name__ == "__main__":
    answer = generation_pipeline(
        query="Je cherche un Data Scientist, trouve une expérience semblable."
    )

    print("Here is the answer:")
    print(answer)