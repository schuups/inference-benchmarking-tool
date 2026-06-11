# FirecREST MCP

1. Setup `.env` credentials file

2. Spin up MCP server

```sh
# From the repo root
source .venv/bin/activate
uv run firecrest-mcp/server.py
```

3. Register the MCP server within Claude.
```sh
claude mcp add firecrest http://localhost:8888/mcp --transport http
```

4. Test
```sh
source .venv/bin/activate
python tools/pre-flight-checks.py
```