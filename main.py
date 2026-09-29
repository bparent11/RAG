from fastapi import FastAPI, UploadFile, File
from src.ingestion import load_resume_from_api, split_resume, create_get_vector_store, add_chunks_to_database


app = FastAPI()

@app.post("/upload-cv")
async def upload_cv(file: UploadFile = File(...)):
    # 1. Lire le contenu du fichier reçu
    content = await file.read()

    document = load_resume_from_api(
        content=content,
        filename=file.filename
    )
    chunks = split_resume(resume=document)
    vectorstore = create_get_vector_store()
    add_chunks_to_database(chunks=chunks, vectorstore=vectorstore)

    # 2. Ici vous mettriez votre logique (extraction texte, LLM, etc.)
    # Pour l'exemple, on renvoie juste une confirmation
    return {
        "text": "Here is the first extracted chunk",
        "chunk": chunks[0].page_content,
    }