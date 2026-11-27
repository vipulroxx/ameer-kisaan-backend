"""Application settings.

Every value comes from environment variables (or a local .env file). In AWS the
ECS task definition injects secrets from Secrets Manager into the same variable
names, so the code never reads secrets from disk or source control.
"""
import os
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    # Tests never read a developer's .env, so they cannot touch real resources.
    model_config = SettingsConfigDict(env_file=None if os.getenv("APP_ENV") == "test" else ".env", extra="ignore")

    app_name: str = "Ameer Kisaan API"
    app_env: Literal["local", "test", "staging", "prod"] = "local"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"

    # --- Auth -----------------------------------------------------------
    jwt_secret: str = Field(min_length=32)
    jwt_issuer: str = "ameer-kisaan"
    jwt_audience: str = "ameer-kisaan-web"
    access_token_ttl_minutes: int = 15
    refresh_token_ttl_days: int = 7
    login_max_attempts: int = 5
    login_lockout_minutes: int = 15
    # 32-byte key, base64 encoded, used for AES-256-GCM field encryption of
    # sensitive identifiers (Aadhaar, PAN). Stored in Secrets Manager in AWS.
    field_encryption_key: str = Field(min_length=40)

    # --- Web / cookies --------------------------------------------------
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000"]  # comma-separated in env
    cookie_secure: bool = True
    max_request_mb: int = 8
    # How to find the client's IP for rate limits and audit logs (see middleware.client_ip).
    trusted_proxy_hops: int = 1
    client_ip_header: str | None = None
    max_upload_mb: int = 5

    # --- AWS --------------------------------------------------------------
    aws_region: str = "ap-south-1"
    dynamodb_endpoint_url: str | None = None   # DynamoDB Local in docker-compose
    s3_endpoint_url: str | None = None         # MinIO in docker-compose
    s3_public_endpoint_url: str | None = None  # host the browser uses for presigned URLs
    s3_bucket: str = "ameer-kisaan-uploads"
    table_prefix: str = "ameer-kisaan-"
    presigned_url_ttl_seconds: int = 900

    # --- PostgreSQL (alternative to DynamoDB, e.g. Supabase free tier) ---------
    # When set, tables live in Postgres via app/db/pgtable.py instead of DynamoDB.
    database_url: str | None = None
    database_pool_size: int = 5
    # "presigned": browsers fetch photos straight from S3 with a presigned URL.
    # "proxy": the API streams them via a short-lived signed link of its own, so
    # any S3-compatible store works (e.g. Supabase Storage) and the bucket stays private.
    photo_url_mode: Literal["presigned", "proxy"] = "presigned"

    # --- Web app ------------------------------------------------------------
    # Folder with the built React app. When set, the API also serves the web
    # app on the same origin (single-service hosting, e.g. Render).
    static_dir: str | None = None

    # --- AI ---------------------------------------------------------------
    llm_provider: Literal["ollama", "openai", "azure_openai", "fake"] = "ollama"
    embeddings_provider: Literal["ollama", "openai", "azure_openai", "fake"] = "ollama"
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_embed_model: str = "nomic-embed-text"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"
    # Optional separate OpenAI-compatible endpoint/key for embeddings, so chat and
    # embeddings can come from different providers (e.g. Groq chat + Gemini embeddings).
    openai_embed_base_url: str | None = None
    openai_embed_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str | None = None
    azure_openai_embed_deployment: str | None = None
    llm_timeout_seconds: float = 60.0
    # Zero-cost guard: hard ceilings per UTC day on calls to cloud AI providers, kept below
    # the free plans' own daily quotas (Groq ~1,000 requests and ~200,000 tokens, Gemini
    # embeddings ~1,000 requests). Past a ceiling the app refuses the call instead of making
    # it, so even if a payment card were ever added to a provider account, usage could never
    # go beyond what the free plan allows. 0 switches that provider kind off entirely.
    ai_daily_chat_requests: int = 900
    ai_daily_chat_tokens: int = 180_000
    ai_daily_embed_requests: int = 900

    knowledge_dir: str = "knowledge"
    index_cache_dir: str = "/tmp/ameer-kisaan-index"  # noqa: S108 - private per-container disk, rebuilt on start
    rag_top_k: int = 4
    rag_max_distance: float = 0.75

    vision_model_path: str = "models/crop_classifier.tflite"
    vision_labels_path: str = "models/crop_classifier.labels.json"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    def table(self, name: str) -> str:
        return f"{self.table_prefix}{name}"

    @property
    def is_prod(self) -> bool:
        return self.app_env in ("staging", "prod")


@lru_cache
def get_settings() -> Settings:
    return Settings()
