import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter # chunking
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma # main reason for using Chroma is that we can host it locally
from dotenv import load_dotenv

load_dotenv()



# llm = OllamaLLM(model="llama3.2:1b")
# embeddings = OllamaEmbeddings(model="nomic-embed-text")


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. PLease create it and add your company files.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader # if we have pdf files we also have classes to handle it (here it is made for .txt)
    )

    documents = loader.load()
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]} ...")
        print(f"metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=512, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n Chunk {i+1} ".center(30, "-"))
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-"*50)

    if len(chunks) > 5:
        print(f"\n ... and {len(chunks) - 5} more chunks.")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents( # vector database created locally
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"} # specifying searching algorithm to be cosine similarity
    )
    print("--- Finished creating vector store ---")
    print(f"Vector store created and saved to {persist_directory}")

    return vectorstore

    # llm = OllamaLLM(model="llama3.2:1b")
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")

def main():
    print("Main Function")

    #1. Loading the files
    documents = load_documents(docs_path="docs")
 
    #2. Chunking the files
    chunks = split_documents(documents)

    #3. Embedding and Storing in Vector DB
    vectorstore = create_vector_store(chunks)


if __name__ == "__main__":
    main()