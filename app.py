import streamlit as st
import requests

st.title("Mon Application RAG - Neofacto")

# 1. Widget pour uploader un CV
uploaded_file = st.file_uploader("Téléchargez un CV (PDF)", type=["pdf"])

if uploaded_file is not None:
    # 2. Bouton pour déclencher l'action
    if st.button("Analyser ce CV via l'API"):
        
        # On prépare le fichier pour qu'il soit envoyé en HTTP
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        with st.spinner("Envoi à l'API en cours..."):
            try:
                # 3. Appel de l'API FastAPI (qui tourne en local sur le port 8000)
                response = requests.post("http://127.0.0.1:8000/upload-cv", files=files)
                
                # 4. Affichage du résultat
                if response.status_code == 200:
                    st.success("Réponse de l'API reçue avec succès !")
                    st.title(response.json()["text"])
                    st.text(response.json()["chunk"])
                else:
                    st.error(f"Erreur de l'API : {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Impossible de contacter l'API. Est-ce que FastAPI est bien lancé ?")


# Discuss with RAG system
chat_input = st.text_input("Discuss with a RAG system", "", type="search")
if chat_input:
    query = {"query":chat_input}

    with st.spinner("Réponse en cours de génération ..."):
        try:
            response = requests.post("http://127.0.0.1:8000/query", params=query)
        
            if response.status_code == 200:
                st.success("Réponse de l'API")
                st.text(response.json()["answer"])
            else:
                st.error(f"Impossible de générer la réponse. Veuillez réessayer plus tard.")
        
        except requests.exceptions.ConnectionError:
            st.error("Impossible de contacter l'API. Est-ce que FastAPI est bien lancé ?")
    