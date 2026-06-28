import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT_DIR))

from rag.rag_engine import RAGEngine

def test_engine_load():

    engine = RAGEngine()

    assert engine.index is not None
    assert engine.metadata is not None
    


def test_retrieve_documents():

    engine = RAGEngine()

    docs = engine.retrieve_documents(
        "concert à Paris"
    )

    assert len(docs) > 0
    


def test_ask():

    engine = RAGEngine()

    response = engine.ask(
        "Je cherche un concert"
    )

    assert isinstance(response, str)
    assert len(response) > 0