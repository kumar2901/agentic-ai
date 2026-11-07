import re

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

from paper_pilot.config import Provider, Settings, get_settings

COLLECTION = "papers"


DEFAULT_EMBEDDDING_MODELS: dict[Provider, str] = {
    "groq": "sentence-transformers/all-MiniLM-L6-v2",
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    "ollama": "nomic-embed-text",
}


def get_embedding_model_name(settings: Settings) -> str:
    """The configured embedding model or providers default"""
    return DEFAULT_EMBEDDDING_MODELS[settings.llm_provider]


def collection_name(settings: Settings) -> str:
    """One collection per embedding model: vectors from different model can't be mized"""
    safe = re.sub(r"[^a-z0-9]+", "-", get_embedding_model_name(settings).lower()).strip("-")
    return f"papers-{safe}"


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Local embedding model: text -> vector of numbers"""
    s = settings or get_settings()
    model_name = get_embedding_model_name(s)
    if s.llm_provider == "ollama":
        # ollama's embeddings API lives at base url without /v1
        return OllamaEmbeddings(model=model_name, base_url=s.ollama_base_url.removesuffix("/v1"))
    return HuggingFaceEmbeddings(model_name=s.embedding_model)


def vector_store(settings: Settings | None = None) -> Chroma:
    """Open or create the chroma collection from saved on disk"""
    s = settings or get_settings()
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(s),
        persist_directory=str(s.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
