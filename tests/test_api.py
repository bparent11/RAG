from fastapi.testclient import TestClient
from unittest.mock import patch
from pathlib import Path
from main import app

client = TestClient(app)

@patch("main.add_chunks_to_database")
@patch("main.create_get_vector_store")
def test_upload_cv_api_route(mock_get_vector, mock_add_chunks):
    """
    Testing FastAPI endpoint
    Mocking database to avoid writing in it
    """
    # On charge un vrai PDF de test pour que le parsing s'exécute pour de vrai
    pdf_path = Path(__file__).parent / "data" / "sample_cv.pdf"
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
        
    # On prépare le fichier pour le TestClient FastAPI
    files = {"file": ("sample_cv.pdf", pdf_bytes, "application/pdf")}
    
    # Appel de l'API
    response = client.post("/upload-cv", files=files)
    
    # Vérifications de l'API
    assert response.status_code == 200
    data = response.json()
    
    # On vérifie que l'API renvoie bien la structure attendue
    assert "text" in data
    assert "chunk" in data
    assert isinstance(data["chunk"], str)
    assert len(data["chunk"]) > 0
    
    # On vérifie que la base de données a bien été appelée par l'endpoint
    mock_get_vector.assert_called_once()
    mock_add_chunks.assert_called_once()