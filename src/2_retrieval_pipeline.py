from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

persisten_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory=persisten_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

# Search for relevant documents
# query = "Which company is related the most to gaming?"
# query = "From which company should I buy a car?"
query = "Tell me about algorithms"


# retriever = db.as_retriever(search_kwargs={"k":3})
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k":5,
        "score_threshold":0.5 # Only return chunks with cosine similarity >= 0.5
    }
)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs):
    print(f"\nRetrieved document {i+1}")
    print(f"Source: {doc.metadata['source']}")
    print(f"Content: {doc.page_content}")

# on peut valider la qualité du RAG en demandant à un autre LLMs si selon la question du user, il y a la réponse dans les résultats.