import os
from pathlib import Path
from src.ingestion import load_resume_from_api, split_resume
from src.retrieval import retrieve_k_similar_docs
from src.generation import generate_first_answer

from langchain_core.documents.base import Document
from dotenv import load_dotenv

load_dotenv()
persistent_dir = os.getenv("PERSISTENT_DIRECTORY")

def test_real_pdf_parsing_and_splitting():
    """Testing ingestion pipeline: reading and chunking of a pdf file"""
    
    # Chemin vers le mini PDF de test
    pdf_path = Path(__file__).parent / "data" / "sample_cv.pdf"
    
    # S'assure que le fichier de test existe bien
    assert pdf_path.exists(), "Le fichier sample_cv.pdf est manquant dans tests/data/"
    
    # 1. Lecture du fichier réel en octets
    with open(pdf_path, "rb") as f:
        file_content = f.read()
        
    # 2. Test de la VRAIE fonction d'ingestion
    document = load_resume_from_api(content=file_content, filename="sample_cv.pdf")
    assert document is not None
    
    # 3. Test du VRAI découpage (splitting)
    chunks = split_resume(resume=document)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert len(chunks[0].page_content) > 0


def test_retrieval_system():
    """Testing retrieval system with a specific query."""
    # asking a question in another language can help not to find any similar chunk.
    k_similar_docs = retrieve_k_similar_docs(
        query="Find the best work experiences for a Data Scientist position.",
        k=2,
        persistent_dir=persistent_dir
    )

    assert isinstance(k_similar_docs, list)
    assert len(k_similar_docs[0].page_content) > 0, "A chunk seems to be empty."

def test_generation_system():
    """Testing generation system after"""
    query = "Find the best work experiences for a Data Scientist position."

    mock_k_similar_docs = Document
    mock_k_similar_docs.page_content = "Don't generate anything."

    answer = generate_first_answer(
        query=query,
        relevant_docs=[mock_k_similar_docs]
    )

    assert isinstance(answer, str)
    assert len(answer) > 0