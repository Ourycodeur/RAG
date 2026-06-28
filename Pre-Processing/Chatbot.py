# Créons le chatbot intelligent
# Importations des modules
import faiss
import pandas as pd
import numpy as np

# Chargeons les indexes
index = faiss.read_index(
    "../Pre-Processing/faiss/event_index.faiss"
)
print(index.ntotal)

# Chargeons les métadonnées
metadata = pd.read_csv(
    "../Pre-Processing/event_metadata.csv"
)

metadata.head()
# Importations de la clé d'API depuis mistral
from mistralai import Mistral
import os
from dotenv import load_dotenv
api_key = MISTRAL_API_KEY

client = Mistral(api_key=api_key)

# Créations des embedding de la réquête
def embed_query(query):

    response = client.embeddings.create(
        model="mistral-embed",
        inputs=[query]
    )

    return np.array(
        [response.data[0].embedding],
        dtype=np.float32
    )

# Definition du nombre document à récuperer par défaut
def retrieve_documents(
    query,
    k=5
):

    query_vector = embed_query(
        query
    )

    distances, indices = index.search(
        query_vector,
        k
    )

    return metadata.iloc[
        indices[0]
    ]
 
# Créations du contexte    
def build_context(results):

    return "\n\n".join(
        results["chunk"]
        .astype(str)
        .tolist()
    )
    
# Création du prompt selon le contexte
def build_prompt(
    question,
    context
):

    return f"""
Tu es un assistant spécialisé dans
les événements culturels.

Tu dois répondre uniquement
à partir du contexte fourni.

Contexte :

{context}

Question :

{question}

Réponse :
"""

# Créations de la réponse 

def generate_answer(
    question
):

    retrieved_docs = retrieve_documents(
        question
    )

    context = build_context(
        retrieved_docs
    )

    prompt = build_prompt(
        question,
        context
    )

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

## Intéractions entre Bot et Humain

while True:

    question = input(
        "\nVous : "
    )

    if question.lower() == "quit":
        break

    answer = generate_answer(
        question
    )

    print(
        "\nBot :",
        answer
    )
