from pathlib import Path
import os, tempfile

from fastapi import UploadFile, File
from langchain_core.documents.base import Document
from langchain_community.document_loaders import TextLoader, PDFPlumberLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

persist_dir = os.environ.get("PERSISTENT_DIRECTORY", "db_chroma_db")

def load_resume_from_local_path(docs_path: str = "data/resume") -> list[Document]:
    """
    Load documents
    Args: docs_path (str) -> resume's folder

    Return: (list[Document]) -> List of retrieved documents
    """
    print(f"Loading documents from {docs_path}")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.pdf",
        loader_cls=PDFPlumberLoader
    )

    resume = loader.load()

    if len(resume) == 0:
        raise FileNotFoundError(f"No .pdf files found in {docs_path}. Please add resume you need to vectorize.")

    return resume

def load_resume_from_api(content:UploadFile, filename:str) -> Document:
    """
    Load resume
    Args: file (UploadFile) -> document posted by the user

    Return (Document)
    """
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    loader = PDFPlumberLoader(tmp_path)
    document = loader.load()

    document[0].metadata["source"] = filename

    return document


def split_resume(resume: list[Document] | Document, chunk_size: int=512, chunk_overlap: int=0) -> list[Document]:
    # TODO: don't split resume but store a summary of the resume, or different experiences.
    """
    Split documents into chunks

    Args: 
        resume (list[Document]) -> document to chunk
        chunk_size (int)
        chunk_overlap (int)

    Return (list[Document]) -> chunks
    """
    print("Splitting documents into chunks")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "."],
        keep_separator=False,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(resume)

    return chunks


def create_get_vector_store(persist_directory: str="db/chroma_db") -> Chroma:
    """
    Create or get vector database and store chunks
    
    Args:
        chunks (list[Document]) -> chunks to vectorize and store
        persist_directory (str) -> location of the database
    """
    print("Creating vector store")
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    existing = Path(persist_directory, "chroma.sqlite3").is_file()

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space":"cosine"}
    )

    if existing:
        print("Vector store returned.")
    else:
        print(f"Vector store created and saved to {persist_directory}")

    return vectorstore

def add_chunks_to_database(chunks: list[Document] | Document, vectorstore: Chroma) -> None:
    vectorstore.add_documents(
        documents=chunks
    )

def ingest():
    resume = load_resume_from_local_path()
    chunks = split_resume(resume)
    vectorstore = create_get_vector_store(persist_directory=persist_dir)
    add_chunks_to_database(chunks=chunks, vectorstore=vectorstore)

if __name__ == "__main__":
    ingest()