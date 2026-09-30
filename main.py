from fastapi import FastAPI, UploadFile, File
from src.ingestion import load_resume_from_api, split_resume, create_get_vector_store, add_chunks_to_database
from src.retrieval import retrieve_k_similar_docs
from src.generation import generate_first_answer

import os
from dotenv import load_dotenv

load_dotenv()
persistent_dir = os.getenv("PERSISTENT_DIRECTORY")

app = FastAPI()

@app.post("/upload-cv")
async def upload_cv(file: UploadFile = File(...)):
    content = await file.read()

    document = load_resume_from_api(
        content=content,
        filename=file.filename
    )
    chunks = split_resume(resume=document)
    vectorstore = create_get_vector_store()
    add_chunks_to_database(chunks=chunks, vectorstore=vectorstore)

    return {
        "text": "Here is the first extracted chunk",
        "chunk": chunks[0].page_content,
    }

@app.post("/query")
async def query_rag_system(query:str):
    relevant_docs = retrieve_k_similar_docs(
        query=query,
        persistent_dir=persistent_dir
        )
    
    answer = generate_first_answer(
        query=query,
        relevant_docs=relevant_docs
        )

    return {
        "answer":answer
    }