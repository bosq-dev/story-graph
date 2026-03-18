from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ALLOWED_ENTITY_TYPES = {
    "User",
    "Company",
    "Product",
    "Technology",
    "Feature",
    "Issue",
    "Activity",
    "Location",
    "Concept",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Story Graph Backend"
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_base_url: str = Field(default="https://api.openai.com/v1", alias="LLM_BASE_URL")
    llm_model: str = Field(default="openai/gpt-4o-mini", alias="LLM_MODEL")
    llm_provider: str = Field(default="", alias="LLM_PROVIDER")

    neo4j_uri: str = Field(default="bolt://neo4j:7687", alias="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field(default="password", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    sqlite_path: str = Field(default="/data/chat_history.db", alias="SQLITE_PATH")
    extraction_confidence_default: float = Field(default=0.75, alias="EXTRACTION_CONFIDENCE_DEFAULT")


@lru_cache
def get_settings() -> Settings:
    return Settings()
