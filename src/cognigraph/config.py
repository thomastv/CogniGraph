import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    llm_base_url: str | None
    openai_api_key: str | None
    tavily_api_key: str | None
    obsidian_vault_path: str | None


def load_settings() -> Settings:
    """Load environment settings from .env and process defaults."""
    load_dotenv()
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        llm_model=os.getenv("LLM_MODEL", "gemma"),
        llm_base_url=os.getenv("LLM_BASE_URL"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        obsidian_vault_path=os.getenv("OBSIDIAN_VAULT_PATH"),
    )
