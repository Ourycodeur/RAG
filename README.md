# Projet 9 - Chatbot RAG de Recommandation d'Événements Culturels

## Description

Ce projet consiste à développer un système RAG (Retrieval Augmented Generation) capable de recommander des événements culturels à partir d'une base de données d'événements indexée dans FAISS.

L'utilisateur pose une question en langage naturel et le système :

1. Recherche les événements les plus pertinents dans la base vectorielle FAISS.
2. Construit un contexte à partir des documents retrouvés.
3. Utilise un modèle Mistral pour générer une réponse naturelle.
4. Retourne une recommandation pertinente à l'utilisateur.

Le projet est accessible via :

* Une API REST FastAPI
* Une interface utilisateur Streamlit
* Une exécution conteneurisée avec Docker

---

# Architecture du projet

```
P9/
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── rag/
│   └── rag_engine.py
│
├── Pre-Processing/
│   ├── data/
│   │   └── event_index.faiss
         └── event_metadata.csv
│
├── tests/
│   ├── test_api.py
│   ├── test_rag_engine.py
│   └── test_evaluate.py
│
├── Mistralchat.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
│
└── README.md
```

---

# Technologies utilisées

* Python
* FastAPI
* Streamlit
* FAISS
* LangChain
* Mistral AI
* Ragas
* Pytest
* Docker
* GitHub Actions

---

# Installation

## Cloner le dépôt

```bash
git clone <https://github.com/Ourycodeur/RAG.git>
cd P9
```

## Créer un environnement virtuel

```bash
python -m venv .venv
```

```

```

## Installer les dépendances

```bash
pip install -r requirements.txt
```

---

# Variables d'environnement

Créer un fichier `.env`

```env
MISTRAL_API_KEY
MODEL_ID
EMBEDDING_MODEL
```

---

# Construction de l'index vectoriel

Le projet utilise :

* event_index.faiss
* event_metadata.csv

Ces fichiers sont générés lors du prétraitement des événements.

---

# Lancement de l'API

```bash
uvicorn api.main:app --reload
```

API disponible sur :

```text
http://localhost:8000
```

Documentation Swagger :

```text
http://localhost:8000/docs
```

---

# Endpoints

## POST /ask

Permet de poser une question au chatbot.

Exemple :

```json
{
  "question": "Je cherche un concert à Paris"
}
```

Réponse :

```json
{
  "answer": "Concert Gospel le 21 juin 2025..."
}
```

---

## POST /rebuild

Permet de reconstruire la base vectorielle.

Réponse :

```json
{
  "message": "Base vectorielle reconstruite"
}
```

---

# Interface Streamlit

Lancer l'application :

```bash
streamlit run MistralChat.py
```

Fonctionnalités :

* Saisie de questions
* Génération de recommandations
* Affichage des réponses du chatbot

---

# Évaluation du système RAG

L'évaluation est réalisée avec Ragas.

Métriques utilisées :

* Faithfulness
* Answer Relevancy
* Context Precision
* Context Recall

Exemple :

```python
result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)
```

---

# Jeu de test annoté

Le projet contient un ensemble de questions et réponses de référence (ground truths).

Exemples :

Question :

```
Je cherche des concerts à Paris.
```

Réponse attendue :

```
Concert Gospel le 21 juin 2025 à Paris.
```

Ces données sont utilisées pour évaluer automatiquement les performances du système.

---

# Tests unitaires

Exécution :

```bash
pytest tests -v
```

Tests réalisés :

## test_rag_engine.py

* Chargement de l'index FAISS
* Chargement des métadonnées
* Vérification des colonnes attendues

## test_api.py

* Validation d'une question vide
* Vérification du endpoint /rebuild

---

# Rapport de couverture

Installation :

```bash
pip install pytest-cov
```

Génération :

```bash
pytest tests -v \
--cov=api \
--cov=rag \
--cov-report=term \
--cov-report=html
```

Rapport HTML :

```text
htmlcov/index.html
```

---

# Docker

## Construction de l'image

```bash
docker build -t p9-rag .
```

## Lancement du conteneur

```bash
docker run -p 8000:8000 p9-rag
```

---

# Docker Compose

```bash
docker compose up --build
```

---

# Intégration Continue (CI)

Le projet utilise GitHub Actions.

Déclenchement :

* Push sur la branche main
Pipeline :

1. Installation des dépendances
2. Exécution des tests unitaires
3. Vérification de la qualité du code

---

# Résultats obtenus

Le système permet :

* La recherche sémantique d'événements
* La génération de réponses augmentées par récupération
* Une interaction via API REST
* Une interaction via Streamlit
* Une évaluation automatique avec Ragas
* Une exécution conteneurisée avec Docker
* Une automatisation des tests avec GitHub Actions

---

# Auteur

Mamadou Oury Baldé

Étudiant en Mathématiques et Informatique

Université Gamal Abdel Nasser de Conakry
