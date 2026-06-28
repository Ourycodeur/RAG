import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT_DIR))
import streamlit as st
from rag.rag_engine import RAGEngine

st.set_page_config(
    page_title="Assistant Événements Culturels pour la ville de Paris",
    page_icon="🎭",
    layout="wide"
)

@st.cache_resource
def load_engine():
    return RAGEngine()

engine = load_engine()

st.title("🎭 Assistant Événements Culturels")
st.caption("Propulsé par Mistral + FAISS + RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Posez votre question..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    answer = engine.ask(question)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
    
with st.sidebar:
    st.header("Informations")

    st.write("Modèle : mistral-large-latest")
    st.write("Base vectorielle : FAISS")
    st.write("Données : OpenAgenda")