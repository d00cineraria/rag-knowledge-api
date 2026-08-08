from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://rag:rag_dev_password@localhost:5432/rag"
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768
    api_dev_key: str = "dev-local-key"
    max_upload_mb: int = 20
    reranker_enabled: bool = False
    data_dir: str = "/data"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
