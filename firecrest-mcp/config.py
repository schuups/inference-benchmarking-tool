# config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Injected by K8s External Secrets from Vault
    oauth_client_id: str
    oauth_client_secret: str
    oauth_token_url: str
    oauth_scopes: str
    backend_api_base_url: str

    class Config:
        env_file = ".env"


settings = Settings()
