from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    sqlite_path: str = "./data/rag.db"
    llm_provider: str = "ollama"  # "ollama" | "gemini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_chat_model: str = "qwen3.5:9b"
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768
    api_dev_key: str = "dev-local-key"
    max_upload_mb: int = 20
    reranker_enabled: bool = False
    data_dir: str = "./data"
    # "inline": process_documentがembeddingまで完結する（既定・従来動作）。
    # "worker": process_documentはchunks/chunks_ftsへのINSERTまでで止め、
    #           status='embedding'とし、embeddingはGoワーカー（worker/）に委ねる。
    ingest_mode: str = "inline"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
