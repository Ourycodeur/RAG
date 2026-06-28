import sys
import os
import pandas as pd
from pathlib import Path
import traceback
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from rag.rag_engine import RAGEngine
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig

engine = RAGEngine()

# ============================================================
# QUESTIONS : ciblées, spécifiques, proches des vraies requêtes
# utilisateurs — éviter les questions trop larges
# ============================================================
questions = [
    "Je cherche des concerts de jazz à Paris.",
    "Quels spectacles de théâtre sont proposés ?",
    "Y a-t-il des mariages qui sont célébrés ?",
]

# ============================================================
# GROUND TRUTHS : construites APRES avoir inspecté ce que
# ton index FAISS contient réellement pour chaque question.
#
# RÈGLE : une ground_truth doit contenir les entités concrètes
# qu'une bonne réponse inclurait — titre, lieu, date, détails.
# Ragas va vérifier claim par claim si le contexte les couvre.
# ============================================================
ground_truths = [
    # Q1 — basé sur ce que le print a montré réellement
    (
        "Un concert de jazz est proposé à Paris : Concert BB15 jazz, "
        "à l'Hôpital Broca, le 21 septembre 2025. "
        "L'événement invite à découvrir les ruines du couvent."
    ),

    # Q2 — le print a confirmé Théâtre vivant à la Cité des sciences
    (
        "Des spectacles de théâtre sont proposés à Paris. "
        "Théâtre vivant est présenté à la Cité des sciences et de l'Industrie "
        "le 21 juin 2025 à 12h00, par la compagnie Specta."
    ),

    # Q3 — le print a confirmé : aucun mariage, RAG a retourné Inalco
    (
        "Aucune cérémonie de mariage n'est référencée dans la base. "
        "Les événements disponibles sont de nature culturelle : "
        "patrimoine, arts textiles, concerts et expositions."
    ),
]

# ============================================================
# COLLECTE DES RÉPONSES ET CONTEXTES
# ============================================================
answers = []
contexts = []

for i, question in enumerate(questions):
    print(f"\n[{i+1}/{len(questions)}] Traitement : {question}")

    response = engine.ask(question)
    answers.append(response)

    docs = engine.retrieve_documents(question)
    context = engine.build_context(docs)
    contexts.append([context])

    # Affichage du contexte récupéré pour valider/ajuster les ground_truths
    print(f"  → Contexte récupéré (extrait) : {context[:200]}...")
    print(f"  → Réponse générée (extrait)   : {response[:200]}...")

    time.sleep(3)  # Rate limit Mistral

# ============================================================
# DATASET RAGAS
# ============================================================
dataset = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
)

# ============================================================
# CONFIGURATION ET ÉVALUATION
# ============================================================
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
if not MISTRAL_API_KEY:
    print("⚠️ AVERTISSEMENT : Clé API Mistral non trouvée.")

try:
    print("\nInitialisation LLM et Embeddings Mistral...")
    from langchain_mistralai import ChatMistralAI
    from langchain_mistralai import MistralAIEmbeddings

    mistral_llm = ChatMistralAI(
        mistral_api_key=MISTRAL_API_KEY,
        model="mistral-large-latest",
        temperature=0.1
    )
    mistral_embeddings = MistralAIEmbeddings(
        mistral_api_key=MISTRAL_API_KEY
    )
    print("LLM et Embeddings initialisés.")

    metrics_to_evaluate = [
        faithfulness,       # Génération : fidèle au contexte ?
        answer_relevancy,   # Génération : réponse pertinente ?
        context_precision,  # Récupération : peu de bruit ?
        context_recall,     # Récupération : infos clés présentes ?
    ]

    print(f"Métriques : {[m.name for m in metrics_to_evaluate]}")

    print("\nLancement évaluation Ragas...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics_to_evaluate,
        llm=mistral_llm,
        embeddings=mistral_embeddings,
        run_config=RunConfig(
            max_workers=1,   # Séquentiel pour éviter les 429
            timeout=180,
            max_retries=5,
            seed = 42
        ),
        raise_exceptions= False
    )
    print("\n--- Évaluation Ragas terminée ---")

    results_df = results.to_pandas()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 150)

    print("\n--- Résultats par question ---")
    print(results_df)

    print("\n--- Scores Moyens ---")
    average_scores = results_df.mean(numeric_only=True)
    print(average_scores)

    # Interprétation automatique des scores
    print("\n--- Interprétation ---")
    thresholds = {
        "faithfulness": 0.8,
        "answer_relevancy": 0.7,
        "context_precision": 0.7,
        "context_recall": 0.6,
    }
    for metric, threshold in thresholds.items():
        if metric in average_scores:
            score = average_scores[metric]
            status = "✅" if score >= threshold else "⚠️"
            print(f"  {status} {metric}: {score:.2f} (seuil: {threshold})")

except Exception as e:
    print(f"\n❌ ERREUR : {e}")
    traceback.print_exc()
    
    