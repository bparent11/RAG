from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

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
# query = "Tell me about algorithms"
query = "Tell me the age of my son."


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
# print("--- Context ---")
# for i, doc in enumerate(relevant_docs):
#     print(f"\nRetrieved document {i+1}")
#     print(f"Source: {doc.metadata['source']}")
#     print(f"Content: {doc.page_content}")

# on peut valider la qualité du RAG en demandant à un autre LLMs si selon la question du user, il y a la réponse dans les résultats.

# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the document (or if there are no documents), say "I don't have enough information to answer that question based on the provided documents.
"""

# Create a model
model = OllamaLLM(model="llama3.2:1b")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)
]

result = model.invoke(messages)

print("--- Generated response ---")
print("Content only:")
print(result)