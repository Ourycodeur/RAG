import os
import faiss
import pandas as pd
import numpy as np
from pathlib import Path
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()
ROOT_DIR = Path(__file__).resolve().parent.parent



class RAGEngine:

    def __init__(self):
        self.load_resources()

    def load_resources(self):

        # Chargement FAISS
        self.index = faiss.read_index(
            str(ROOT_DIR / "Pre-Processing" / "data" / "event_index.faiss")
        )

        # Chargement metadata
        self.metadata = pd.read_csv(
            str(ROOT_DIR / "Pre-Processing" / "data" / "event_metadata.csv")
        )
        ## code pour l'api
        # Client Mistral
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        api_key = MISTRAL_API_KEY

        ## Defi
        self.client = Mistral(
            api_key=api_key
        )

        print("✅ Ressources chargées")
        print(f"📚 Documents : {len(self.metadata)}")
        print(f"🔎 Index FAISS : {self.index.ntotal}")

    def embed_query(self, question):

        response = self.client.embeddings.create(
            model="mistral-embed",
            inputs=[question]
        )

        embedding = response.data[0].embedding

        return np.array(
            [embedding],
            dtype=np.float32
        )

    def retrieve_documents(
        self,
        question,
        k=5
    ):

        query_vector = self.embed_query(
            question
        )

        distances, indices = self.index.search(
            query_vector,
            k
        )

        results = self.metadata.iloc[
            indices[0]
        ]

        return results

    def build_context(
        self,
        results
    ):

        context = ""

        for _, row in results.iterrows():

            context += f"""
Titre : {row['Titre']}

Ville : {row['Ville']}

Lieu : {row['Lieu']}

Date : {row['Date']}

Description :
{row['chunk']}

------------------------------------
"""

        return context

    def build_prompt(
        self,
        question,
        context
    ):

        return f"""
Tu es un assistant spécialisé dans les événements culturels.

Ta mission est de recommander les événements les plus pertinents.

Réponds uniquement à partir du contexte fourni.

Si aucune information pertinente n'est présente,
indique clairement que tu ne disposes pas de la réponse.

CONTEXTE :

{context}

QUESTION :

{question}

RÉPONSE :
"""

    def generate_answer(
        self,
        question
    ):

        documents = self.retrieve_documents(
            question,
            k=5
        )

        context = self.build_context(
            documents
        )

        prompt = self.build_prompt(
            question,
            context
        )

        response = self.client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    def ask(
        self,
        question
    ):

        return self.generate_answer(
            question
        )

    def rebuild(self):

        print(
            "Reconstruction de l'index demandée"
        )

        return {
            "status": "success",
            "message": "Base vectorielle reconstruite"
        }