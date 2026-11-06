from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from paper_pilot.config import Settings, get_settings

DEFAULT_MODEL = {
    "groq": "openai/got-oss-20b",
    "ollama": "mistral:latest",
    "huggingface": "meta-llama/Llama-3.1-8B-Instruct",
}


def get_model(settings: Settings | None = None) -> ChatOpenAI:
    """Get the model based on the settings."""
    s = settings or get_settings()
    model_name = s.llm_model or DEFAULT_MODEL[s.llm_provider]

    if s.llm_provider == "groq":
        base_url, api_key = (
            "https://api.groq.com/openai/v1",
            _require(s.groq_api_key, "GROQ_API_KEY"),
        )
    elif s.llm_provider == "huggingface":
        base_url, api_key = (
            "https://router.huggingface.co/v1",
            _require(s.hugging_face_api_key, "HUGGING_FACE_API_KEY"),
        )
    else:
        base_url, api_key = s.ollama_base_url, "ollama"

    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        temperature=s.llm_temperature,
        timeout=s.llm_timeout_seconds,
        max_retries=1,
    )


def _require(secret: SecretStr | None, name: str) -> str:
    """Require a secret value to be present."""
    if secret is None:
        raise ValueError(f"{name} must be set in the environment.")
    return secret.get_secret_value()
