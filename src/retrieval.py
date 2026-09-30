import os

from langchain_core.documents.base import Document

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from dotenv import load_dotenv

load_dotenv()

persistent_dir = os.environ.get("PERSISTENT_DIRECTORY", "db_chroma_db")
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

def retrieve_k_similar_docs(query: str, k: int = 5, persistent_dir:str = persistent_dir) -> list[Document] | str:
    db = Chroma(
        persist_directory=persistent_dir,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space":"cosine"}
    )

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k":5,
            "score_threshold":0.5
        }
    )

    relevant_docs = retriever.invoke(query)

    if relevant_docs != []:
        print(f"{len(relevant_docs)} docs found!")
        return relevant_docs
    
    empty_doc = Document
    empty_doc.page_content = "Any reliable source found."

    return [empty_doc]

if __name__ == "__main__":
    relevant_docs = retrieve_k_similar_docs(
        query="Je cherche un Data Scientist, trouve une expérience semblable."
    )

    print("Here is the most relevant document:")
    print(relevant_docs[0].page_content)