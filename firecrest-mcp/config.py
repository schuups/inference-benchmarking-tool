# config.py
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # FirecREST OAuth2 client-credentials + API base for ONE platform.
    oauth_client_id: str
    oauth_client_secret: str
    oauth_token_url: str
    oauth_scopes: str
    backend_api_base_url: str
    # No default `env_file` on purpose — see load_settings(): the path is always
    # passed explicitly, so a stray ./.env in the cwd is never picked up.


def load_settings(env_file: str) -> Settings:
    """Load credentials from an explicit ``.env`` file path.

    There is intentionally **no default ``.env``**: each FirecREST platform runs as its
    own MCP server instance with its own credentials file — ML Platform for
    ``clariden`` / ``bristen``, HPC Platform for ``beverin`` — so the path must be given
    explicitly (``server.py --env-file``). Raises if the file does not exist.
    """
    path = Path(env_file).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"--env-file not found: {path}")
    return Settings(_env_file=str(path))
