import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT_DIR))
from rag.rag_engine import RAGEngine

def test_engine_load():
    engine = RAGEngine()

    assert engine.index is not None
    

def test_metadata_load():
    engine = RAGEngine()

    assert engine.metadata is not None
    assert len(engine.metadata) > 0


def test_retrieve_documents():
    engine = RAGEngine()

    docs = engine.retrieve_documents(
        "concert à Paris"
    )

    assert len(docs) > 0