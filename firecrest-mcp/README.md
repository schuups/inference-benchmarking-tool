# FirecREST MCP

Run **one server instance per FirecREST platform**, each pointed at its own credentials
file via `--env-file`. There is **no default `.env`** — the path is always explicit, so
the two platforms never share or accidentally pick up the wrong credentials:

| Instance | Covers |
|---|---|
| ML Platform  | `clariden`, `bristen` |
| HPC Platform | `beverin` |

1. Create a credentials `.env` per platform (kept out of git — `.env.*` is ignored), each
   with: `oauth_client_id`, `oauth_client_secret`, `oauth_token_url`, `oauth_scopes`,
   `backend_api_base_url`. e.g. `~/.firecrest/mlp.env` and `~/.firecrest/hpc.env`.

2. Spin up one server per platform — required `--env-file`, distinct `--port`:

```sh
# From the repo root
source .venv/bin/activate
uv run firecrest-mcp/server.py --env-file ~/.firecrest/mlp.env --port 8888   # clariden, bristen
uv run firecrest-mcp/server.py --env-file ~/.firecrest/hpc.env --port 8889   # beverin
```

3. Register each within Claude:
```sh
claude mcp add firecrest-mlp http://localhost:8888/mcp --transport http   # ML Platform (clariden, bristen)
claude mcp add firecrest-hpc http://localhost:8889/mcp --transport http   # HPC Platform (beverin)
```

4. Test
```sh
source .venv/bin/activate
python tools/pre-flight-checks.py
```
