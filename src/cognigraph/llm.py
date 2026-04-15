import logging

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from cognigraph.config import Settings


def get_llm(settings: Settings):
    """Initialize and return the configured LLM client."""
    logging.info(f"Initializing LLM with provider: {settings.llm_provider}")

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            logging.error("OPENAI_API_KEY is not set.")
            raise ValueError("OPENAI_API_KEY is not set in the .env file.")
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            temperature=0,
        )

    if not settings.llm_base_url:
        logging.error("LLM_BASE_URL is not set for Ollama.")
        raise ValueError("LLM_BASE_URL is required for the 'ollama' provider.")

    logging.info(f"Using Ollama model: {settings.llm_model} from {settings.llm_base_url}")
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        temperature=0,
    )
