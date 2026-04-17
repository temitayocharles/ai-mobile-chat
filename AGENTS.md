# AGENTS.md — ai-mobile-chat (Jerry)

## What this repo is

`ai-mobile-chat` is **Jerry** — a private AI assistant chat application for TCA InfraForge.
It runs as a macOS LaunchAgent on the operator's Mac (port 8088) and is accessible externally
via Cloudflare Tunnel at `ai.tca-infraforge.site` (behind Cloudflare Access).

**Purpose:** Personal AI assistant for homelab operations, infrastructure queries, code help,
and file/PDF analysis. Primarily used from iPhone (mobile PWA) and desktop browser.

**Tech stack:** Python FastAPI, SQLite WAL, Ollama local LLMs, SSE streaming, vanilla JS PWA

## Architecture

```
app.py          — single-file FastAPI backend + inline HTML/CSS/JS frontend (~3700 lines)
data/           — legacy DB location (migrated to ~/Library/Application Support/ai-mobile-chat/)
requirements.txt

Runtime:
  DB:       ~/Library/Application Support/ai-mobile-chat/jerry_chat.db
  Service:  ~/Library/LaunchAgents/site.tca-infraforge.ai-mobile-chat.plist
  Venv:     ~/.venvs/ai-mobile-chat/
  Logs:     ~/Library/Logs/ai-mobile-chat.{out,err}.log
```

## Models

| Model | Use |
|-------|-----|
| `qwen3.5:2b-q4_K_M` | Fast responses (default) |
| `qwen3.5:4b-q4_K_M` | Quality responses + tool-calling |
| `qwen3-vl:2b-instruct-q4_K_M` | Vision / image analysis |
| `nomic-embed-text` | Memory embeddings |

All models run via local Ollama on port 11434.

## Operating boundaries

- Never hardcode API keys, tokens, or secrets in `app.py`
- Never expose internal infrastructure details (cluster IPs, node names) via the public endpoint
- The `propose_write` tool enforces policy gates from `policy/jerry_command_policy.json`:
  - `delete` / `edit` classes are denied
  - `restart` / `sync` classes are auto-approved
  - other write classes are ask-first approval
- `run_command` is READ-ONLY (kubectl get/describe/logs, df, ps, curl) — never use for mutations
- Do not change `DATA_DIR` from `~/Library/Application Support/ai-mobile-chat/` back to repo-relative path

## Key areas of the codebase

- `BASE_SYSTEM_PROMPT` (~line 201) — Jerry's identity and tool discipline rules
- `get_db_connection()` (~line 285) — context manager, always closes connection
- `execute_tool()` (~line 1125) — all tool implementations (run_command, prometheus_query, etc.)
- `policy/jerry_command_policy.json` — deterministic allow/deny/ask policy source of truth
- `run_tool_loop()` (~line 1307) — shared tool loop used by both streaming and non-streaming paths
- `event_stream()` (~line 1743) — SSE streaming handler
- `renderMd()` / `startStreamingBubble()` — frontend markdown + streaming UI

## Service management

```bash
# Restart
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/site.tca-infraforge.ai-mobile-chat.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/site.tca-infraforge.ai-mobile-chat.plist

# Health check
curl -s http://127.0.0.1:8088/health | python3 -m json.tool

# Logs
tail -f ~/Library/Logs/ai-mobile-chat.err.log
```

## Development workflow

- Test syntax before restarting: `python3 -m py_compile app.py`
- After any edit: restart service and hit `/health`
- Frontend changes: hard-refresh the browser (Cmd+Shift+R) to clear cached HTML
- Repo: `temitayocharles/ai-mobile-chat` on GitHub, branch `main`

## Web resources

Use webfetch and websearch freely for library docs, API references, and current versions.
