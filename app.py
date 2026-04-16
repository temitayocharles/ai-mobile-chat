import asyncio
import base64
import json
import os
import re
import sqlite3
import subprocess
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Literal, Optional

import fitz
import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
TEXT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b-q4_K_M")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:2b-instruct-q4_K_M")

MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "14"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))

RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "12"))
MAX_USER_MESSAGE_CHARS = int(os.getenv("MAX_USER_MESSAGE_CHARS", "4000"))
MAX_ATTACHMENTS_PER_MESSAGE = int(os.getenv("MAX_ATTACHMENTS_PER_MESSAGE", "4"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(4 * 1024 * 1024)))
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", str(10 * 1024 * 1024)))
MAX_PDF_TEXT_CHARS = int(os.getenv("MAX_PDF_TEXT_CHARS", "20000"))
TOOL_COMMAND_TIMEOUT = int(os.getenv("TOOL_COMMAND_TIMEOUT", "15"))
PROMETHEUS_BASE_URL = os.getenv("PROMETHEUS_BASE_URL", "http://kube-prometheus-stack-prometheus.observability.svc.cluster.local:9090")
LOKI_BASE_URL = os.getenv("LOKI_BASE_URL", "http://loki.observability.svc.cluster.local:3100")
# Dedicated embedding model — chat models don't support embeddings
# Pull with: ollama pull nomic-embed-text
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Patterns that indicate a mutating/write operation — must go through propose_write approval gate
WRITE_COMMAND_PATTERNS = [
    r"\bkubectl\s+(delete|patch|edit|apply|replace|scale|set|label|annotate|taint|cordon|uncordon|drain|expose|create|run)\b",
    r"\bkubectl\s+rollout\s+(restart|undo)\b",
    r"\bkubectl\s+exec\b",
    r"\bkubectl\s+cp\b",
    r"\bhelm\s+(install|upgrade|uninstall|rollback|delete)\b",
    r"\bargocd\s+(app\s+set|app\s+delete|app\s+sync|repo\s+add|repo\s+rm|cluster\s+add|cluster\s+rm)\b",
    r"\bgit\s+(push|commit|reset|rebase|merge|force)\b",
    r"\bvault\s+(write|delete|kv\s+put|kv\s+delete|policy\s+write|auth\s+enable)\b",
    r"\bsystemctl\s+(start|stop|restart|enable|disable)\b",
    r"\bapt(-get)?\s+(install|remove|purge)\b",
    r"\byum\s+(install|remove|erase)\b",
    r"\bpip\s+install\b",
    r"\bnpm\s+(install|uninstall)\b",
]

ALLOWED_IMAGE_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
}

BLOCKED_COMMAND_PATTERNS = [
    "rm -rf", "rm -f /", "dd if=", "mkfs", "> /dev/",
    "shutdown", "reboot", "halt", "poweroff",
    ":(){ :|:& };:",
]

JERRY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a READ-ONLY shell command to inspect infrastructure state. "
                "Use for: kubectl get/describe/logs/top, helm list/status, argocd app get, "
                "ping, curl GET, df, free, ps, journalctl -r, cat/less/grep on files, etc. "
                "DO NOT use for any mutating operation (delete, apply, patch, scale, restart, install, etc.) — "
                "use propose_write for those instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The read-only shell command to execute."}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prometheus_query",
            "description": (
                "Query Prometheus metrics using PromQL instant query. "
                "Use for CPU usage, memory consumption, error rates, request rates, pod restarts, "
                "or any metrics-based question. Returns metric labels and values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "PromQL expression (e.g. 'rate(http_requests_total[5m])')."},
                    "time": {"type": "string", "description": "Optional RFC3339 or Unix timestamp. Defaults to now."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "loki_query",
            "description": (
                "Query Loki logs using LogQL. Use for fetching recent errors, application logs, "
                "or searching log patterns. Supports label matchers and line filters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "LogQL query (e.g. '{namespace=\"production\",app=\"myapp\"} |= \"error\"')."},
                    "since": {"type": "string", "description": "Look-back duration (e.g. 1h, 30m, 24h). Default: 30m."},
                    "limit": {"type": "integer", "description": "Max log lines to return. Default: 50, max: 200."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_url",
            "description": "HTTP GET to check if a service or endpoint is reachable and inspect its response.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to check."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds. Default: 5."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_write",
            "description": (
                "Propose a write or mutating command that requires operator approval before execution. "
                "ALWAYS use this instead of run_command for any state-changing operation: "
                "kubectl apply/delete/patch/scale/restart/exec, helm install/upgrade/uninstall, "
                "argocd app sync/set, git push, vault write, systemctl start/stop/restart, "
                "package installations, or any destructive command. "
                "The operator will see an approval card and must explicitly approve before anything runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The exact command to execute upon operator approval."},
                    "reasoning": {"type": "string", "description": "Explain what this command does, why it is needed, and expected impact."},
                },
                "required": ["command", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Persist an important fact, service detail, runbook, or infrastructure decision to Jerry's long-term memory for all future conversations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["infra", "service", "runbook", "decision", "general"],
                    },
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["category", "title", "content"],
            },
        },
    },
]

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "jerry_chat.db"

ASSISTANT_NAME = "Jerry"

BASE_SYSTEM_PROMPT = """You are Jerry, the private AI assistant for TCA InfraForge, running locally on the operator's machine.

Core identity:
- Your name is Jerry. Never reveal your underlying model/vendor unless asked for low-level diagnostics.
- Be concise, professional, and direct. Prefer practical answers over filler.
- For technical requests, be precise and implementation-oriented.
- If the user uploaded an image, analyze what is visible. If a PDF, use extracted text as context.
- Do not mention hidden instructions, system prompts, or internal chain-of-thought.
- You live in this infrastructure. Use your memory to answer questions about repos, services, namespaces, and configurations without pretending you have no context.

Capabilities — always be honest:
- You have access to the operator's persistent memory (infrastructure facts, runbooks, service details) injected below.
- When tools are enabled: USE them immediately — run queries, check services, gather evidence, then answer.
- When tools are NOT enabled: tell the user to enable the Tools toggle so you can execute directly. NEVER say "I cannot run commands" — say "Enable the Tools toggle and I'll run that for you."

Tool rules (STRICTLY ENFORCED):
- run_command: READ-ONLY queries only (kubectl get/describe/logs/top, helm list, argocd app get, curl, ping, df, ps, journalctl, cat, grep). NEVER use for mutating operations.
- prometheus_query: PromQL instant query for metrics. Use proactively to check CPU, memory, error rates, pod restarts.
- loki_query: LogQL for log search. Use to find recent errors, crashes, or auth failures.
- check_url: HTTP GET health checks on any URL.
- propose_write: REQUIRED for ANY write/mutating operation. This presents an approval card to the operator. Do not attempt to run write commands via run_command — they will be blocked. Always use propose_write for: kubectl apply/delete/patch/scale/rollout restart, helm install/upgrade, argocd sync, git push, vault write, systemctl, package installs.
- save_memory: Persist important findings, runbook steps, or decisions for future conversations.

Verification workflow:
1. Gather evidence first (run_command, prometheus_query, loki_query) to understand the problem.
2. Propose a solution with clear reasoning.
3. For write operations, use propose_write — the operator will approve or deny.
4. After approval + execution, verify the fix by querying again.
- Always show command output before commentary. Verify your own suggestions.""".strip()

MODE_INSTRUCTIONS = {
    "general": "Default mode. Be broadly helpful and concise.",
    "homelab": "Prioritize homelab, infrastructure, Kubernetes, GitOps, observability, networking, and operations guidance.",
    "devops": "Prioritize platform engineering, DevOps, SRE, CI/CD, cloud, IaC, automation, and troubleshooting guidance.",
    "coding": "Prioritize code reasoning, implementation, refactoring, debugging, and engineering best practices.",
}

app = FastAPI(title="TCA InfraForge AI Mobile Chat")

rate_limit_store: dict[str, deque] = defaultdict(deque)
rate_limit_lock = Lock()


class AttachmentInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    mime_type: str = Field(..., min_length=1, max_length=255)
    data_base64: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    conversation_id: Optional[int] = None
    mode: Literal["general", "homelab", "devops", "coding"] = "general"
    message: str = Field("", max_length=MAX_USER_MESSAGE_CHARS)
    attachments: list[AttachmentInput] = Field(default_factory=list)
    think: bool = False
    tools_enabled: bool = False


class ProfileUpdateRequest(BaseModel):
    preferred_name: str = Field("", max_length=100)


class MemoryCreateRequest(BaseModel):
    category: Literal["infra", "service", "runbook", "decision", "general"] = "general"
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=4000)


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_db_connection() -> sqlite3.Connection:
    ensure_data_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profile (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                preferred_name TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO profile (id, preferred_name)
            VALUES (1, '')
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'general',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                latency_ms INTEGER,
                attachment_meta_json TEXT,
                model_used TEXT,
                tool_events_json TEXT,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL DEFAULT 'general',
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_writes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                command TEXT NOT NULL,
                reasoning TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                output TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        # Migrations: add columns introduced after initial schema
        existing_mem_cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
        if "embedding_json" not in existing_mem_cols:
            conn.execute("ALTER TABLE memories ADD COLUMN embedding_json TEXT")
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        if "model_used" not in existing_cols:
            conn.execute("ALTER TABLE messages ADD COLUMN model_used TEXT")
        if "attachment_meta_json" not in existing_cols:
            conn.execute("ALTER TABLE messages ADD COLUMN attachment_meta_json TEXT")
        if "latency_ms" not in existing_cols:
            conn.execute("ALTER TABLE messages ADD COLUMN latency_ms INTEGER")
        if "tool_events_json" not in existing_cols:
            conn.execute("ALTER TABLE messages ADD COLUMN tool_events_json TEXT")
        existing_conv_cols = {row[1] for row in conn.execute("PRAGMA table_info(conversations)")}
        if "mode" not in existing_conv_cols:
            conn.execute("ALTER TABLE conversations ADD COLUMN mode TEXT NOT NULL DEFAULT 'general'")
        # FTS5 full-text search
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content, content=messages, content_rowid=id)"
            )
            conn.execute(
                """CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
                END"""
            )
            conn.execute(
                """CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
                    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
                END"""
            )
            conn.execute(
                """CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
                END"""
            )
        except Exception:
            pass
        conn.commit()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_preferred_name() -> str:
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT preferred_name FROM profile WHERE id = 1"
        ).fetchone()
    if not row:
        return ""
    return (row["preferred_name"] or "").strip()


def set_preferred_name(name: str) -> str:
    cleaned = " ".join(name.strip().split())
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE profile SET preferred_name = ? WHERE id = 1",
            (cleaned,),
        )
        conn.commit()
    return cleaned


def reset_preferred_name() -> None:
    with get_db_connection() as conn:
        conn.execute("UPDATE profile SET preferred_name = '' WHERE id = 1")
        conn.commit()


def list_memories() -> list[dict]:
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, category, title, content, created_at, updated_at FROM memories ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def create_memory(category: str, title: str, content: str) -> int:
    t = now_iso()
    with get_db_connection() as conn:
        cur = conn.execute(
            "INSERT INTO memories (category, title, content, created_at, updated_at) VALUES (?,?,?,?,?)",
            (category, title.strip(), content.strip(), t, t),
        )
        conn.commit()
        mem_id = int(cur.lastrowid)
    # Generate and store embedding so semantic search works for this memory
    embedding = get_embedding_sync(f"{category}: {title}. {content}")
    if embedding:
        store_embedding(mem_id, embedding)
    return mem_id


def delete_memory(memory_id: int) -> bool:
    with get_db_connection() as conn:
        cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Semantic memory search — uses Ollama /api/embed for vector embeddings
# so Jerry understands generic/colloquial queries without exact keyword matching
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_embedding_sync(text: str) -> list[float] | None:
    """Call Ollama /api/embed synchronously and return the embedding vector."""
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": text},
            )
            data = resp.json()
            embeddings = data.get("embeddings", [])
            if embeddings and isinstance(embeddings[0], list):
                return embeddings[0]
            return None
    except Exception:
        return None


async def get_embedding(text: str) -> list[float] | None:
    """Async version of get_embedding_sync."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": TEXT_MODEL, "input": text},
            )
            data = resp.json()
            embeddings = data.get("embeddings", [])
            if embeddings and isinstance(embeddings[0], list):
                return embeddings[0]
            return None
    except Exception:
        return None


def store_embedding(memory_id: int, embedding: list[float]) -> None:
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE memories SET embedding_json = ? WHERE id = ?",
            (json.dumps(embedding), memory_id),
        )
        conn.commit()


def semantic_search_memories(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Return the top-k most semantically similar memories to the query embedding."""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, category, title, content, embedding_json FROM memories"
        ).fetchall()
    scored = []
    for row in rows:
        if not row["embedding_json"]:
            continue
        try:
            emb = json.loads(row["embedding_json"])
        except Exception:
            continue
        score = _cosine_similarity(query_embedding, emb)
        scored.append((score, dict(row)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:top_k]]


def build_system_prompt(mode: str) -> str:
    preferred_name = get_preferred_name()
    prompt = BASE_SYSTEM_PROMPT + f"\n\nMode instruction: {MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS['general'])}"
    if preferred_name:
        prompt += (
            f"\n\nUser profile:"
            f"\n- The user's preferred name is {preferred_name}."
            f"\n- Address the user as {preferred_name} when natural and appropriate."
        )
    mems = list_memories()
    if mems:
        lines = [f"[{m['category']}] {m['title']}: {m['content']}" for m in mems]
        prompt += "\n\nJerry's persistent infrastructure memory:\n" + "\n".join(f"- {l}" for l in lines)
    return prompt


def build_system_prompt_with_semantic_context(mode: str, user_query: str) -> str:
    """
    Like build_system_prompt but uses semantic similarity to inject the most
    relevant memories for this specific query rather than all memories.
    Falls back to all memories if embeddings are unavailable.
    """
    preferred_name = get_preferred_name()
    prompt = BASE_SYSTEM_PROMPT + f"\n\nMode instruction: {MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS['general'])}"
    if preferred_name:
        prompt += (
            f"\n\nUser profile:"
            f"\n- The user's preferred name is {preferred_name}."
            f"\n- Address the user as {preferred_name} when natural and appropriate."
        )
    # Try semantic retrieval first
    query_emb = get_embedding_sync(user_query)
    if query_emb:
        relevant = semantic_search_memories(query_emb, top_k=8)
        if relevant:
            lines = [f"[{m['category']}] {m['title']}: {m['content']}" for m in relevant]
            prompt += "\n\nMost relevant infrastructure memory (semantically matched):\n" + "\n".join(f"- {l}" for l in lines)
            return prompt
    # Fallback: inject all memories
    mems = list_memories()
    if mems:
        lines = [f"[{m['category']}] {m['title']}: {m['content']}" for m in mems]
        prompt += "\n\nJerry's persistent infrastructure memory:\n" + "\n".join(f"- {l}" for l in lines)
    return prompt


def search_messages(query: str, limit: int = 30) -> list[dict]:
    if not query.strip():
        return []
    with get_db_connection() as conn:
        try:
            rows = conn.execute("""
                SELECT m.id, m.conversation_id, m.role, m.content, m.created_at,
                       c.title AS conversation_title
                FROM messages_fts fts
                JOIN messages m ON m.id = fts.rowid
                JOIN conversations c ON c.id = m.conversation_id
                WHERE messages_fts MATCH ?
                ORDER BY rank LIMIT ?
            """, (query, limit)).fetchall()
        except Exception:
            rows = conn.execute("""
                SELECT m.id, m.conversation_id, m.role, m.content, m.created_at,
                       c.title AS conversation_title
                FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                WHERE m.content LIKE ? ORDER BY m.id DESC LIMIT ?
            """, (f"%{query}%", limit)).fetchall()
    return [dict(r) for r in rows]


def get_client_ip(request: Request) -> str:
    x_forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def get_authenticated_identity(request: Request) -> str:
    header_candidates = [
        "cf-access-authenticated-user-email",
        "x-auth-request-email",
    ]
    for header_name in header_candidates:
        value = request.headers.get(header_name, "").strip()
        if value:
            return value
    return ""


def enforce_rate_limit(client_id: str) -> dict:
    now = time.time()
    with rate_limit_lock:
        bucket = rate_limit_store[client_id]
        while bucket and (now - bucket[0]) > RATE_LIMIT_WINDOW_SECONDS:
            bucket.popleft()

        if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
            retry_after = int(max(1, RATE_LIMIT_WINDOW_SECONDS - (now - bucket[0])))
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again in about {retry_after} seconds.",
            )

        bucket.append(now)
        remaining = max(0, RATE_LIMIT_MAX_REQUESTS - len(bucket))
        return {
            "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
            "max_requests": RATE_LIMIT_MAX_REQUESTS,
            "current_requests": len(bucket),
            "remaining_requests": remaining,
        }


def current_rate_limit_snapshot(client_id: str) -> dict:
    now = time.time()
    with rate_limit_lock:
        bucket = rate_limit_store[client_id]
        while bucket and (now - bucket[0]) > RATE_LIMIT_WINDOW_SECONDS:
            bucket.popleft()
        return {
            "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
            "max_requests": RATE_LIMIT_MAX_REQUESTS,
            "current_requests": len(bucket),
            "remaining_requests": max(0, RATE_LIMIT_MAX_REQUESTS - len(bucket)),
        }


def summarize_title(message: str) -> str:
    stripped = " ".join(message.strip().split())
    if not stripped:
        return "New chat"
    return stripped[:60] + ("..." if len(stripped) > 60 else "")


def create_conversation(initial_message: str, mode: str) -> int:
    current_time = now_iso()
    title = summarize_title(initial_message)
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO conversations (title, mode, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (title, mode, current_time, current_time),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_conversation(conversation_id: int) -> sqlite3.Row | None:
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
    return row


def list_conversations() -> list[dict]:
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, title, mode, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC
            """
        ).fetchall()

    return [
        {
            "id": int(row["id"]),
            "title": row["title"],
            "mode": row["mode"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]


def save_message(
    conversation_id: int,
    role: str,
    content: str,
    latency_ms: Optional[int] = None,
    attachment_meta: Optional[list[dict]] = None,
    model_used: Optional[str] = None,
    tool_events: Optional[list[dict]] = None,
) -> int:
    current_time = now_iso()
    attachment_meta_json = json.dumps(attachment_meta or [], ensure_ascii=False)
    tool_events_json = json.dumps(tool_events or [], ensure_ascii=False)
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO messages (conversation_id, role, content, created_at, latency_ms, attachment_meta_json, model_used, tool_events_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (conversation_id, role, content, current_time, latency_ms, attachment_meta_json, model_used, tool_events_json),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (current_time, conversation_id),
        )
        conn.commit()
        return int(cursor.lastrowid)


def update_assistant_message(message_id: int, content: str, latency_ms: int, model_used: str, tool_events: Optional[list[dict]] = None) -> None:
    tool_events_json = json.dumps(tool_events or [], ensure_ascii=False)
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE messages
            SET content = ?, latency_ms = ?, model_used = ?, tool_events_json = ?
            WHERE id = ?
            """,
            (content, latency_ms, model_used, tool_events_json, message_id),
        )
        conn.commit()


def update_conversation_title(conversation_id: int, title: str) -> None:
    with get_db_connection() as conn:
        conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title[:120], conversation_id))
        conn.commit()


def fetch_messages(conversation_id: int) -> list[dict]:
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, role, content, created_at, latency_ms, attachment_meta_json, model_used, tool_events_json
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

    result = []
    for row in rows:
        try:
            attachment_meta = json.loads(row["attachment_meta_json"] or "[]")
        except json.JSONDecodeError:
            attachment_meta = []
        try:
            tool_events = json.loads(row["tool_events_json"] or "[]")
        except Exception:
            tool_events = []

        result.append(
            {
                "id": int(row["id"]),
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
                "latency_ms": row["latency_ms"],
                "attachments": attachment_meta,
                "model_used": row["model_used"],
                "tool_events": tool_events,
            }
        )
    return result


def delete_conversation(conversation_id: int) -> None:
    with get_db_connection() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()


def decode_attachment_data(data_base64: str) -> bytes:
    try:
        return base64.b64decode(data_base64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 attachment data.") from exc


def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Failed to read PDF file.") from exc

    chunks: list[str] = []
    current_len = 0
    try:
        for page in doc:
            text = page.get_text("text").strip()
            if not text:
                continue
            remaining = MAX_PDF_TEXT_CHARS - current_len
            if remaining <= 0:
                break
            text = text[:remaining]
            chunks.append(text)
            current_len += len(text)
    finally:
        doc.close()

    return "\n\n".join(chunks).strip()


def prepare_current_turn_content(
    message_text: str,
    attachments: list[AttachmentInput],
) -> tuple[str, list[str], list[dict]]:
    if len(attachments) > MAX_ATTACHMENTS_PER_MESSAGE:
        raise HTTPException(
            status_code=400,
            detail=f"Too many attachments. Maximum is {MAX_ATTACHMENTS_PER_MESSAGE}.",
        )

    content = message_text.strip()
    image_payloads: list[str] = []
    attachment_meta: list[dict] = []
    pdf_sections: list[str] = []

    for attachment in attachments:
        file_bytes = decode_attachment_data(attachment.data_base64)
        mime = attachment.mime_type.lower()
        name = attachment.name.strip()

        if mime == "application/pdf":
            if len(file_bytes) > MAX_PDF_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"PDF '{name}' exceeds the maximum size limit.",
                )
            pdf_text = extract_pdf_text(file_bytes)
            if not pdf_text:
                pdf_text = "[No extractable text found in PDF.]"

            pdf_sections.append(f"[PDF attachment: {name}]\n{pdf_text}")
            attachment_meta.append(
                {
                    "name": name,
                    "mime_type": mime,
                    "kind": "pdf",
                }
            )

        elif mime == "image/svg+xml":
            raise HTTPException(
                status_code=400,
                detail=(
                    f"SVG uploads are not supported for image analysis right now. "
                    f"Please upload a PNG, JPG, or WebP version of '{name}'."
                ),
            )

        elif mime.startswith("image/"):
            if mime not in ALLOWED_IMAGE_MIME_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported image type for '{name}'. Use PNG, JPG, or WebP.",
                )

            if len(file_bytes) > MAX_IMAGE_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image '{name}' exceeds the maximum size limit.",
                )

            image_payloads.append(base64.b64encode(file_bytes).decode("utf-8"))
            attachment_meta.append(
                {
                    "name": name,
                    "mime_type": mime,
                    "kind": "image",
                }
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported attachment type for '{name}'. Only PNG/JPG/WebP images and PDFs are supported.",
            )

    if pdf_sections:
        content = (
            content
            + ("\n\n" if content else "")
            + "Use the following PDF content as context:\n\n"
            + "\n\n---\n\n".join(pdf_sections)
        ).strip()

    if not content and not image_payloads:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    return content, image_payloads, attachment_meta


def select_model_for_request(image_payloads: list[str]) -> str:
    return VISION_MODEL if image_payloads else TEXT_MODEL


def build_ollama_messages(
    conversation_id: int,
    mode: str,
    current_user_content: str,
    current_user_images: list[str],
) -> list[dict]:
    history = fetch_messages(conversation_id)
    history = history[-MAX_HISTORY_MESSAGES:]

    # Use semantic context retrieval when a text query is available (no images)
    if current_user_content and not current_user_images:
        system_content = build_system_prompt_with_semantic_context(mode, current_user_content)
    else:
        system_content = build_system_prompt(mode)

    ollama_messages: list[dict] = [
        {"role": "system", "content": system_content}
    ]

    for msg in history:
        if msg["role"] not in {"user", "assistant"}:
            continue
        ollama_messages.append(
            {
                "role": msg["role"],
                "content": msg["content"],
            }
        )

    current_msg = {
        "role": "user",
        "content": current_user_content,
    }
    if current_user_images:
        current_msg["images"] = current_user_images

    ollama_messages.append(current_msg)
    return ollama_messages


def get_conversation_payload(conversation_id: int) -> dict:
    conversation = get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    return {
        "id": int(conversation["id"]),
        "title": conversation["title"],
        "mode": conversation["mode"],
        "created_at": conversation["created_at"],
        "updated_at": conversation["updated_at"],
        "messages": fetch_messages(conversation_id),
    }


def export_conversation_as_text(conversation: dict) -> str:
    lines = [
        f"Title: {conversation['title']}",
        f"Mode: {conversation['mode']}",
        f"Created: {conversation['created_at']}",
        f"Updated: {conversation['updated_at']}",
        "",
    ]

    for msg in conversation["messages"]:
        who = "You" if msg["role"] == "user" else "Jerry"
        lines.append(f"{who} [{msg['created_at']}]")
        lines.append(msg["content"])
        if msg.get("attachments"):
            for att in msg["attachments"]:
                lines.append(f"[Attachment: {att['name']} | {att['kind']}]")
        if msg.get("latency_ms") is not None:
            lines.append(f"[Latency: {msg['latency_ms']} ms]")
        if msg.get("model_used"):
            lines.append(f"[Model: {msg['model_used']}]")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def export_conversation_as_markdown(conversation: dict) -> str:
    lines = [
        f"# {conversation['title']}",
        "",
        f"- Mode: `{conversation['mode']}`",
        f"- Created: `{conversation['created_at']}`",
        f"- Updated: `{conversation['updated_at']}`",
        "",
    ]

    for msg in conversation["messages"]:
        who = "You" if msg["role"] == "user" else "Jerry"
        lines.append(f"## {who}")
        lines.append("")
        lines.append(msg["content"])
        lines.append("")
        if msg.get("attachments"):
            for att in msg["attachments"]:
                lines.append(f"- Attachment: **{att['name']}** (`{att['kind']}`)")
            lines.append("")
        if msg.get("latency_ms") is not None:
            lines.append(f"- Latency: `{msg['latency_ms']} ms`")
        if msg.get("model_used"):
            lines.append(f"- Model: `{msg['model_used']}`")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def is_command_safe(cmd: str) -> tuple[bool, str]:
    cmd_lower = cmd.lower()
    for pattern in BLOCKED_COMMAND_PATTERNS:
        if pattern.lower() in cmd_lower:
            return False, f"Command blocked: contains '{pattern}'"
    return True, ""


def is_write_command(cmd: str) -> bool:
    """Detect mutating/write operations that require operator approval."""
    for pattern in WRITE_COMMAND_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return True
    return False


def store_pending_write(command: str, reasoning: str, conversation_id: int | None = None) -> int:
    t = now_iso()
    with get_db_connection() as conn:
        cur = conn.execute(
            "INSERT INTO pending_writes (conversation_id, command, reasoning, status, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (conversation_id, command, reasoning, "pending", t, t),
        )
        conn.commit()
        return int(cur.lastrowid)


def execute_tool(name: str, args: dict, conversation_id: int | None = None) -> str:
    if name == "run_command":
        cmd = args.get("command", "").strip()
        if not cmd:
            return "Error: no command provided."
        safe, reason = is_command_safe(cmd)
        if not safe:
            return f"Error: {reason}"
        if is_write_command(cmd):
            return (
                "WRITE_BLOCKED: This is a mutating command and cannot be run directly. "
                "Use the propose_write tool with this command so the operator can review and approve it. "
                f"Command that needs approval: {cmd}"
            )
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=TOOL_COMMAND_TIMEOUT,
                env={**os.environ, "KUBECONFIG": os.environ.get("KUBECONFIG", os.path.expanduser("~/.kube/config"))},
            )
            output = (result.stdout + result.stderr).strip()
            return output[:4000] if output else "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {TOOL_COMMAND_TIMEOUT}s."
        except Exception as exc:
            return f"Error: {exc}"

    elif name == "prometheus_query":
        query = args.get("query", "").strip()
        if not query:
            return "Error: no PromQL query provided."
        params: dict = {"query": query}
        if args.get("time"):
            params["time"] = args["time"]
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(f"{PROMETHEUS_BASE_URL}/api/v1/query", params=params)
            data = resp.json()
            if data.get("status") != "success":
                return f"Prometheus error: {data.get('error', 'unknown error')}"
            result_data = data.get("data", {}).get("result", [])
            if not result_data:
                return "No data returned for this PromQL query."
            lines = [f"Query: {query}"]
            for r in result_data[:25]:
                metric = r.get("metric", {})
                value = r.get("value", ["", ""])
                lines.append(f"  {metric}: {value[1]}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error querying Prometheus ({PROMETHEUS_BASE_URL}): {exc}"

    elif name == "loki_query":
        query = args.get("query", "").strip()
        if not query:
            return "Error: no LogQL query provided."
        since = args.get("since", "30m")
        limit = min(int(args.get("limit", 50)), 200)
        try:
            import time as _time
            now_ns = int(_time.time() * 1e9)
            duration_map = {"m": 60, "h": 3600, "d": 86400}
            unit = since[-1].lower() if since else "m"
            try:
                value = int(since[:-1])
                delta_ns = value * duration_map.get(unit, 60) * int(1e9)
            except (ValueError, IndexError):
                delta_ns = 30 * 60 * int(1e9)
            start_ns = now_ns - delta_ns
            params = {
                "query": query,
                "start": str(start_ns),
                "end": str(now_ns),
                "limit": limit,
                "direction": "backward",
            }
            with httpx.Client(timeout=15) as client:
                resp = client.get(f"{LOKI_BASE_URL}/loki/api/v1/query_range", params=params)
            data = resp.json()
            streams = data.get("data", {}).get("result", [])
            if not streams:
                return "No log entries found for this LogQL query."
            lines = [f"Loki query: {query} (last {since}, limit {limit})"]
            count = 0
            for stream in streams:
                labels = stream.get("stream", {})
                for ts, log_line in stream.get("values", []):
                    lines.append(f"[{labels.get('namespace','?')}/{labels.get('app', labels.get('pod','?'))}] {log_line[:300]}")
                    count += 1
                    if count >= limit:
                        break
                if count >= limit:
                    break
            return "\n".join(lines)
        except Exception as exc:
            return f"Error querying Loki ({LOKI_BASE_URL}): {exc}"

    elif name == "check_url":
        import urllib.request
        url = args.get("url", "").strip()
        timeout = int(args.get("timeout", 5))
        if not url:
            return "Error: no URL provided."
        try:
            req = urllib.request.Request(url, method="GET")
            req.add_header("User-Agent", "Jerry-InfraForge/1.0")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read(512).decode("utf-8", errors="replace")
                return f"HTTP {resp.status} {resp.reason}\nBody preview: {body[:300]}"
        except Exception as exc:
            return f"Unreachable: {exc}"

    elif name == "propose_write":
        command = args.get("command", "").strip()
        reasoning = args.get("reasoning", "").strip()
        if not command:
            return "Error: command is required."
        safe, reason = is_command_safe(command)
        if not safe:
            return f"Error: {reason}"
        pending_id = store_pending_write(command, reasoning, conversation_id)
        return f"APPROVAL_PENDING:{pending_id}"

    elif name == "save_memory":
        category = args.get("category", "general")
        title    = args.get("title", "").strip()
        content  = args.get("content", "").strip()
        if not title or not content:
            return "Error: title and content required."
        mem_id = create_memory(category, title, content)
        return f"Memory saved (id={mem_id}): [{category}] {title}"

    return f"Error: unknown tool '{name}'."


async def run_tool_loop(
    messages: list[dict],
    model: str,
    think: bool,
) -> tuple[str, list[dict]]:
    tool_events: list[dict] = []
    for _ in range(6):
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": model, "messages": messages, "stream": False, "think": think, "tools": JERRY_TOOLS},
            )
            resp.raise_for_status()
            data = resp.json()
        msg = data.get("message", {})
        tool_calls = msg.get("tool_calls", [])
        if not tool_calls:
            content = msg.get("content", "")
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content, tool_events
        messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": tool_calls})
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            result = execute_tool(name, args)
            tool_events.append({"tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "content": result})
    return "Reached maximum tool-use rounds without a final answer.", tool_events


async def generate_smart_title(conversation_id: int, user_msg: str, assistant_msg: str) -> None:
    prompt = (
        f"Write a short title (4-6 words, no quotes, no punctuation at end) for this exchange:\n"
        f"User: {user_msg[:200]}\nAssistant: {assistant_msg[:200]}\nTitle:"
    )
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": TEXT_MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False, "think": False},
            )
            resp.raise_for_status()
            title = resp.json().get("message", {}).get("content", "").strip().strip('"\'').strip()
            if 3 <= len(title) <= 100:
                with get_db_connection() as conn:
                    conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))
                    conn.commit()
    except Exception:
        pass


@app.on_event("startup")
async def startup_event() -> None:
    init_db()
    # Backfill embeddings for any memories that don't have one yet
    asyncio.create_task(_backfill_embeddings())


async def _backfill_embeddings() -> None:
    """Generate embeddings for memories that don't have one yet. Runs at startup."""
    await asyncio.sleep(3)  # Wait for Ollama to be ready
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, category, title, content FROM memories WHERE embedding_json IS NULL"
        ).fetchall()
    for row in rows:
        text = f"{row['category']}: {row['title']}. {row['content']}"
        try:
            emb = await get_embedding(text)
            if emb:
                store_embedding(int(row["id"]), emb)
        except Exception:
            pass  # Non-fatal — system works without embeddings, uses all-memories fallback


@app.get("/health")
async def health() -> dict:
    mems = list_memories()
    return {
        "status": "ok",
        "assistant_name": ASSISTANT_NAME,
        "ollama_base_url": OLLAMA_BASE_URL,
        "text_model": TEXT_MODEL,
        "vision_model": VISION_MODEL,
        "preferred_name": get_preferred_name(),
        "memory_count": len(mems),
        "rate_limit_window_seconds": RATE_LIMIT_WINDOW_SECONDS,
        "rate_limit_max_requests": RATE_LIMIT_MAX_REQUESTS,
        "max_user_message_chars": MAX_USER_MESSAGE_CHARS,
        "allowed_upload_types": ["image/png", "image/jpeg", "image/webp", "application/pdf"],
        "max_image_bytes": MAX_IMAGE_BYTES,
        "max_pdf_bytes": MAX_PDF_BYTES,
    }


@app.get("/api/session")
async def session_info(request: Request) -> dict:
    return {
        "identity": get_authenticated_identity(request),
        "access_protected": True,
    }


@app.get("/api/diagnostics")
async def diagnostics(request: Request) -> dict:
    return {
        "assistant_name": ASSISTANT_NAME,
        "text_model": TEXT_MODEL,
        "vision_model": VISION_MODEL,
        "health": {
            "status": "ok",
            "ollama_base_url": OLLAMA_BASE_URL,
        },
        "upload_limits": {
            "max_user_message_chars": MAX_USER_MESSAGE_CHARS,
            "max_attachments_per_message": MAX_ATTACHMENTS_PER_MESSAGE,
            "max_image_bytes": MAX_IMAGE_BYTES,
            "max_pdf_bytes": MAX_PDF_BYTES,
            "allowed_image_mime_types": sorted(ALLOWED_IMAGE_MIME_TYPES),
        },
        "rate_limit": current_rate_limit_snapshot(get_client_ip(request)),
    }


@app.get("/api/profile")
async def get_profile() -> dict:
    return {"preferred_name": get_preferred_name()}


@app.post("/api/profile")
async def update_profile(payload: ProfileUpdateRequest) -> dict:
    updated_name = set_preferred_name(payload.preferred_name)
    return {"preferred_name": updated_name}


@app.post("/api/profile/reset")
async def reset_profile() -> dict:
    reset_preferred_name()
    return {"preferred_name": ""}


@app.get("/api/conversations")
async def api_list_conversations() -> dict:
    return {"conversations": list_conversations()}


@app.post("/api/conversations")
async def api_create_conversation() -> dict:
    conversation_id = create_conversation("New chat", "general")
    return {
        "conversation_id": conversation_id,
        "conversation": get_conversation_payload(conversation_id),
    }


@app.get("/api/conversations/{conversation_id}")
async def api_get_conversation(conversation_id: int) -> dict:
    return {"conversation": get_conversation_payload(conversation_id)}


@app.delete("/api/conversations/{conversation_id}")
async def api_delete_conversation(conversation_id: int) -> dict:
    if not get_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    delete_conversation(conversation_id)
    return {"deleted": True}


@app.get("/api/conversations/{conversation_id}/export.txt")
async def export_txt(conversation_id: int) -> PlainTextResponse:
    conversation = get_conversation_payload(conversation_id)
    content = export_conversation_as_text(conversation)
    return PlainTextResponse(
        content=content,
        headers={
            "Content-Disposition": f'attachment; filename="conversation-{conversation_id}.txt"'
        },
    )


@app.get("/api/conversations/{conversation_id}/export.md")
async def export_md(conversation_id: int) -> PlainTextResponse:
    conversation = get_conversation_payload(conversation_id)
    content = export_conversation_as_markdown(conversation)
    return PlainTextResponse(
        content=content,
        headers={
            "Content-Disposition": f'attachment; filename="conversation-{conversation_id}.md"'
        },
    )


@app.get("/api/memory")
async def api_list_memories() -> dict:
    return {"memories": list_memories()}


@app.post("/api/memory")
async def api_create_memory(payload: MemoryCreateRequest) -> dict:
    mem_id = create_memory(payload.category, payload.title, payload.content)
    return {"id": mem_id, "category": payload.category, "title": payload.title, "content": payload.content}


@app.delete("/api/memory/{memory_id}")
async def api_delete_memory(memory_id: int) -> dict:
    if not delete_memory(memory_id):
        raise HTTPException(404, "Memory not found.")
    return {"deleted": True}


@app.get("/api/search")
async def api_search(q: str = "", limit: int = 20) -> dict:
    results = search_messages(q, limit=min(limit, 50))
    return {"query": q, "results": results}


@app.post("/api/writes/{write_id}/approve")
async def approve_write(write_id: int) -> JSONResponse:
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM pending_writes WHERE id = ?", (write_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Write proposal not found.")
    if row["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"Write is already {row['status']}.")
    cmd = row["command"]
    safe, reason = is_command_safe(cmd)
    if not safe:
        raise HTTPException(status_code=400, detail=f"Command blocked: {reason}")
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=TOOL_COMMAND_TIMEOUT,
            env={**os.environ, "KUBECONFIG": os.environ.get("KUBECONFIG", os.path.expanduser("~/.kube/config"))},
        )
        output = (result.stdout + result.stderr).strip()[:4000] or "(no output)"
    except subprocess.TimeoutExpired:
        output = f"Command timed out after {TOOL_COMMAND_TIMEOUT}s."
    except Exception as exc:
        output = f"Execution error: {exc}"
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE pending_writes SET status='approved', output=?, updated_at=? WHERE id=?",
            (output, now_iso(), write_id),
        )
        conn.commit()
    return JSONResponse({"status": "approved", "output": output, "command": cmd})


@app.post("/api/writes/{write_id}/deny")
async def deny_write(write_id: int) -> JSONResponse:
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM pending_writes WHERE id = ? AND status = 'pending'", (write_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Pending write not found.")
        conn.execute(
            "UPDATE pending_writes SET status='denied', updated_at=? WHERE id=?",
            (now_iso(), write_id),
        )
        conn.commit()
    return JSONResponse({"status": "denied", "command": row["command"]})


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request, background_tasks: BackgroundTasks) -> JSONResponse:
    client_id = get_client_ip(request)
    enforce_rate_limit(client_id)

    if len(payload.message or "") > MAX_USER_MESSAGE_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Message exceeds {MAX_USER_MESSAGE_CHARS} characters.",
        )

    mode = payload.mode
    conversation_id = payload.conversation_id

    if conversation_id is not None and not get_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")

    current_user_content, current_user_images, attachment_meta = prepare_current_turn_content(
        payload.message,
        payload.attachments,
    )

    if conversation_id is None:
        conversation_id = create_conversation(payload.message or "New chat", mode)

    with get_db_connection() as conn:
        conn.execute(
            "UPDATE conversations SET mode = ?, updated_at = ? WHERE id = ?",
            (mode, now_iso(), conversation_id),
        )
        conn.commit()

    user_content_for_storage = (payload.message or "").strip()
    if attachment_meta:
        summary_lines = [f"[Attachment: {item['name']} | {item['kind']}]" for item in attachment_meta]
        user_content_for_storage = (
            user_content_for_storage
            + ("\n" if user_content_for_storage else "")
            + "\n".join(summary_lines)
        ).strip()

    save_message(
        conversation_id=conversation_id,
        role="user",
        content=user_content_for_storage or "[Empty message with attachments]",
        attachment_meta=attachment_meta,
        model_used=None,
    )

    ollama_messages = build_ollama_messages(
        conversation_id=conversation_id,
        mode=mode,
        current_user_content=current_user_content,
        current_user_images=current_user_images,
    )

    selected_model = select_model_for_request(current_user_images)
    started_at = time.perf_counter()
    tool_events: list[dict] = []

    try:
        if payload.tools_enabled and not current_user_images:
            answer, tool_events = await run_tool_loop(ollama_messages, selected_model, payload.think)
        else:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": selected_model,
                        "messages": ollama_messages,
                        "stream": False,
                        "think": payload.think,
                    },
                )
                response.raise_for_status()
                data = response.json()
            raw_answer = data["message"]["content"]
            answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {exc}") from exc

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)

    save_message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        latency_ms=elapsed_ms,
        attachment_meta=[],
        model_used=selected_model,
        tool_events=tool_events,
    )

    # Smart title on first assistant response
    msg_count = len(fetch_messages(conversation_id))
    if msg_count == 2:
        background_tasks.add_task(generate_smart_title, conversation_id, user_content_for_storage, answer)

    conversation_payload = get_conversation_payload(conversation_id)

    return JSONResponse(
        {
            "conversation_id": conversation_id,
            "conversation": conversation_payload,
            "answer": answer,
            "latency_ms": elapsed_ms,
            "model_used": selected_model,
        }
    )


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request, background_tasks: BackgroundTasks) -> StreamingResponse:
    client_id = get_client_ip(request)
    enforce_rate_limit(client_id)

    if len(payload.message or "") > MAX_USER_MESSAGE_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Message exceeds {MAX_USER_MESSAGE_CHARS} characters.",
        )

    mode = payload.mode
    conversation_id = payload.conversation_id

    if conversation_id is not None and not get_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")

    current_user_content, current_user_images, attachment_meta = prepare_current_turn_content(
        payload.message,
        payload.attachments,
    )

    if conversation_id is None:
        conversation_id = create_conversation(payload.message or "New chat", mode)

    with get_db_connection() as conn:
        conn.execute(
            "UPDATE conversations SET mode = ?, updated_at = ? WHERE id = ?",
            (mode, now_iso(), conversation_id),
        )
        conn.commit()

    user_content_for_storage = (payload.message or "").strip()
    if attachment_meta:
        summary_lines = [f"[Attachment: {item['name']} | {item['kind']}]" for item in attachment_meta]
        user_content_for_storage = (
            user_content_for_storage
            + ("\n" if user_content_for_storage else "")
            + "\n".join(summary_lines)
        ).strip()

    save_message(
        conversation_id=conversation_id,
        role="user",
        content=user_content_for_storage or "[Empty message with attachments]",
        attachment_meta=attachment_meta,
        model_used=None,
    )

    ollama_messages = build_ollama_messages(
        conversation_id=conversation_id,
        mode=mode,
        current_user_content=current_user_content,
        current_user_images=current_user_images,
    )

    selected_model = select_model_for_request(current_user_images)
    assistant_message_id = save_message(
        conversation_id=conversation_id,
        role="assistant",
        content="",
        latency_ms=None,
        attachment_meta=[],
        model_used=selected_model,
    )

    # Capture for closure
    _conv_id = conversation_id
    _user_content = user_content_for_storage

    async def event_stream():
        started_at = time.perf_counter()
        full_answer = ""
        full_thinking = ""
        tool_events: list[dict] = []

        # Tools mode: run tool loop first, then stream final answer
        if payload.tools_enabled and not current_user_images:
            try:
                msgs_copy = list(ollama_messages)
                for _ in range(6):
                    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                        resp = await client.post(
                            f"{OLLAMA_BASE_URL}/api/chat",
                            json={"model": selected_model, "messages": msgs_copy, "stream": False, "think": payload.think, "tools": JERRY_TOOLS},
                        )
                        resp.raise_for_status()
                        data = resp.json()
                    msg = data.get("message", {})
                    tc_list = msg.get("tool_calls", [])
                    if not tc_list:
                        raw = msg.get("content", "")
                        full_answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                        break
                    msgs_copy.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": tc_list})
                    for tc in tc_list:
                        fn = tc.get("function", {})
                        tname = fn.get("name", "")
                        targs = fn.get("arguments", {})
                        yield f"data: {json.dumps({'type': 'tool_call', 'name': tname, 'args': targs}, ensure_ascii=False)}\n\n"
                        result = execute_tool(tname, targs, conversation_id=_conv_id)
                        tool_events.append({"tool": tname, "args": targs, "result": result})
                        # Emit approval_request event for propose_write so UI can render an approval card
                        if tname == "propose_write" and result.startswith("APPROVAL_PENDING:"):
                            pending_id = result.split(":", 1)[1]
                            yield f"data: {json.dumps({'type': 'approval_request', 'id': pending_id, 'command': targs.get('command', ''), 'reasoning': targs.get('reasoning', '')}, ensure_ascii=False)}\n\n"
                        yield f"data: {json.dumps({'type': 'tool_result', 'name': tname, 'result': result}, ensure_ascii=False)}\n\n"
                        msgs_copy.append({"role": "tool", "content": result})
                else:
                    full_answer = "Reached maximum tool-use rounds without a final answer."

                # Emit the answer as tokens
                for ch in full_answer:
                    yield f"data: {json.dumps({'type': 'token', 'token': ch}, ensure_ascii=False)}\n\n"

            except Exception as exc:
                update_assistant_message(
                    message_id=assistant_message_id,
                    content="[Tool loop failed.]",
                    latency_ms=None,
                    model_used=selected_model,
                    tool_events=tool_events,
                )
                yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)}, ensure_ascii=False)}\n\n"
                return

        else:
            # Normal streaming with thinking support
            try:
                in_thinking = False
                thinking_buf = ""

                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                        "POST",
                        f"{OLLAMA_BASE_URL}/api/chat",
                        json={
                            "model": selected_model,
                            "messages": ollama_messages,
                            "stream": True,
                            "think": payload.think,
                        },
                    ) as response:
                        response.raise_for_status()

                        async for line in response.aiter_lines():
                            if await request.is_disconnected():
                                break

                            if not line:
                                continue

                            try:
                                chunk = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            piece = chunk.get("message", {}).get("content", "")
                            if piece:
                                # Parse thinking tags
                                i = 0
                                while i < len(piece):
                                    if not in_thinking:
                                        think_start = piece.find("<think>", i)
                                        if think_start == -1:
                                            normal = piece[i:]
                                            if normal:
                                                full_answer += normal
                                                yield f"data: {json.dumps({'type': 'token', 'token': normal}, ensure_ascii=False)}\n\n"
                                            i = len(piece)
                                        else:
                                            normal = piece[i:think_start]
                                            if normal:
                                                full_answer += normal
                                                yield f"data: {json.dumps({'type': 'token', 'token': normal}, ensure_ascii=False)}\n\n"
                                            in_thinking = True
                                            i = think_start + len("<think>")
                                    else:
                                        think_end = piece.find("</think>", i)
                                        if think_end == -1:
                                            thought = piece[i:]
                                            full_thinking += thought
                                            thinking_buf += thought
                                            yield f"data: {json.dumps({'type': 'thinking', 'token': thought}, ensure_ascii=False)}\n\n"
                                            i = len(piece)
                                        else:
                                            thought = piece[i:think_end]
                                            full_thinking += thought
                                            thinking_buf += thought
                                            if thought:
                                                yield f"data: {json.dumps({'type': 'thinking', 'token': thought}, ensure_ascii=False)}\n\n"
                                            in_thinking = False
                                            i = think_end + len("</think>")

                            if chunk.get("done") is True:
                                break

            except httpx.HTTPError as exc:
                update_assistant_message(
                    message_id=assistant_message_id,
                    content="[Streaming failed before completion.]",
                    latency_ms=None,
                    model_used=selected_model,
                    tool_events=[],
                )
                yield f"data: {json.dumps({'type': 'error', 'detail': f'Ollama request failed: {exc}'}, ensure_ascii=False)}\n\n"
                return

        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        update_assistant_message(
            message_id=assistant_message_id,
            content=full_answer,
            latency_ms=elapsed_ms,
            model_used=selected_model,
            tool_events=tool_events,
        )

        # Smart title on first response
        msg_count = len(fetch_messages(_conv_id))
        if msg_count == 2:
            asyncio.ensure_future(generate_smart_title(_conv_id, _user_content, full_answer))

        conversation_payload = get_conversation_payload(_conv_id)
        done_payload = {
            "type": "done",
            "conversation_id": _conv_id,
            "conversation": conversation_payload,
            "latency_ms": elapsed_ms,
            "model_used": selected_model,
        }
        if full_thinking:
            done_payload["thinking"] = full_thinking
        yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <meta name="mobile-web-app-capable" content="yes" />
  <meta name="apple-mobile-web-app-capable" content="yes" />
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
  <meta name="theme-color" content="#000000" />
  <title>Jerry \u2014 TCA InfraForge AI</title>
  <style>
    /* \u2500\u2500 Reset & base \u2500\u2500 */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html { height: 100%; height: 100dvh; }
    body {
      height: 100%; height: 100dvh;
      background: #000; color: #f2f2f7;
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
      display: flex; flex-direction: column; overflow: hidden;
      overscroll-behavior: none; -webkit-text-size-adjust: 100%;
    }

    /* \u2500\u2500 Header \u2500\u2500 */
    .header {
      flex-shrink: 0;
      background: rgba(28,28,30,0.95);
      backdrop-filter: blur(20px) saturate(180%);
      -webkit-backdrop-filter: blur(20px) saturate(180%);
      border-bottom: 1px solid rgba(255,255,255,0.1);
      padding: env(safe-area-inset-top,0px) env(safe-area-inset-right,0px) 0 env(safe-area-inset-left,0px);
      display: flex; flex-direction: column; position: relative; z-index: 20;
    }
    .header-inner { display: flex; align-items: center; gap: 12px; padding: 10px 16px; min-height: 56px; }
    .header-avatar {
      width: 40px; height: 40px; border-radius: 50%;
      background: linear-gradient(135deg,#0a84ff,#30d158);
      display: flex; align-items: center; justify-content: center;
      font-size: 17px; font-weight: 700; color: #fff; flex-shrink: 0;
    }
    .header-info { flex: 1; min-width: 0; }
    .header-name { font-size: 16px; font-weight: 600; letter-spacing: -0.2px; }
    .header-status { font-size: 12px; color: #8e8e93; margin-top: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .header-btn {
      background: rgba(255,255,255,0.1); border: none; border-radius: 20px;
      color: #0a84ff; font-size: 14px; font-weight: 600;
      padding: 0 16px; height: 44px; cursor: pointer;
      touch-action: manipulation; -webkit-tap-highlight-color: transparent;
      transition: background 0.15s; white-space: nowrap;
    }
    .header-btn:active { background: rgba(255,255,255,0.2); }

    /* \u2500\u2500 Settings drawer \u2500\u2500 */
    .settings-drawer {
      flex-shrink: 0;
      background: rgba(18,18,20,0.99);
      border-bottom: 1px solid rgba(255,255,255,0.08);
      display: none; flex-direction: column; gap: 12px;
      padding: 14px 16px 16px;
      padding-left: max(16px,env(safe-area-inset-left));
      padding-right: max(16px,env(safe-area-inset-right));
      overflow-y: auto; -webkit-overflow-scrolling: touch;
      max-height: 60dvh; max-height: 60vh; overscroll-behavior: contain;
    }
    .settings-drawer.open { display: flex; }
    .settings-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .s-input, .s-select {
      flex: 1; min-width: 0;
      background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12);
      border-radius: 12px; color: #f2f2f7; font-size: 16px;
      padding: 10px 12px; outline: none;
      -webkit-appearance: none; appearance: none;
    }
    .s-select { min-height: 44px; }
    .s-btn {
      border: none; border-radius: 12px; font-size: 14px; font-weight: 600;
      padding: 0 16px; cursor: pointer; min-height: 44px; white-space: nowrap;
      touch-action: manipulation; -webkit-tap-highlight-color: transparent;
      transition: opacity 0.15s; user-select: none; -webkit-user-select: none;
    }
    .s-btn:active { opacity: 0.75; }
    .s-blue { background: #0a84ff; color: #fff; }
    .s-grey { background: rgba(255,255,255,0.12); color: #f2f2f7; }
    .s-red  { background: #ff3b30; color: #fff; }
    .s-green { background: #30d158; color: #fff; }
    .s-label { font-size: 12px; color: #8e8e93; width: 100%; letter-spacing: 0.04em; text-transform: uppercase; }
    .s-diag {
      background: rgba(255,255,255,0.06); border-radius: 12px; padding: 12px;
      font-size: 12px; font-family: ui-monospace,monospace; color: #98989d;
      white-space: pre-wrap; max-height: 180px; overflow-y: auto;
      -webkit-overflow-scrolling: touch;
    }

    /* Toggle rows */
    .s-toggle-row {
      display: flex; align-items: center; justify-content: space-between;
      padding: 2px 0; cursor: pointer; user-select: none;
      font-size: 14px; color: #f2f2f7; width: 100%;
    }
    .s-toggle-row input[type=checkbox] {
      width: 44px; height: 26px; -webkit-appearance: none; appearance: none;
      background: #3a3a3c; border-radius: 13px; cursor: pointer;
      position: relative; transition: background 0.2s; flex-shrink: 0;
    }
    .s-toggle-row input[type=checkbox]:checked { background: #30d158; }
    .s-toggle-row input[type=checkbox]::after {
      content: ''; position: absolute;
      width: 22px; height: 22px; border-radius: 50%; background: #fff;
      top: 2px; left: 2px; transition: left 0.2s;
      box-shadow: 0 1px 3px rgba(0,0,0,0.4);
    }
    .s-toggle-row input[type=checkbox]:checked::after { left: 20px; }

    /* Memory items */
    .memory-item {
      display: flex; align-items: flex-start; gap: 8px;
      padding: 8px; background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08); border-radius: 10px;
    }
    .memory-item-body { flex: 1; min-width: 0; }
    .memory-cat { font-size: 10px; color: #30d158; text-transform: uppercase; letter-spacing: 0.05em; }
    .memory-title { font-size: 13px; color: #f2f2f7; font-weight: 600; margin: 2px 0; }
    .memory-content { font-size: 12px; color: #8e8e93; line-height: 1.4; }
    .memory-del { background: none; border: none; color: #ff453a; font-size: 18px; cursor: pointer; padding: 0 2px; flex-shrink: 0; line-height: 1; }

    /* Search results */
    .search-results { display: flex; flex-direction: column; gap: 6px; margin-top: 6px; }
    .search-result {
      padding: 8px; background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; cursor: pointer;
    }
    .search-result:active { background: rgba(255,255,255,0.08); }
    .search-result-conv { font-size: 10px; color: #636366; margin-bottom: 3px; }
    .search-result-snippet { font-size: 12px; color: #aeaeb2; line-height: 1.4; }

    /* \u2500\u2500 Chat area \u2500\u2500 */
    .chat-area {
      flex: 1; overflow-y: auto; -webkit-overflow-scrolling: touch;
      overscroll-behavior: contain; padding: 12px 12px 6px;
      padding-left: max(12px,env(safe-area-inset-left));
      padding-right: max(12px,env(safe-area-inset-right));
      display: flex; flex-direction: column; gap: 2px; scroll-behavior: smooth;
    }
    .chat-area::-webkit-scrollbar { width: 3px; }
    .chat-area::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 3px; }

    /* \u2500\u2500 Welcome \u2500\u2500 */
    .welcome { margin: auto; text-align: center; color: #8e8e93; padding: 32px 24px; max-width: 320px; }
    .welcome .w-avatar {
      width: 72px; height: 72px; border-radius: 50%;
      background: linear-gradient(135deg,#0a84ff,#30d158);
      display: flex; align-items: center; justify-content: center;
      font-size: 32px; font-weight: 700; color: #fff; margin: 0 auto 16px;
    }
    .welcome h2 { font-size: 20px; font-weight: 700; color: #f2f2f7; margin-bottom: 8px; }
    .welcome p  { font-size: 15px; line-height: 1.55; }

    /* \u2500\u2500 Message groups \u2500\u2500 */
    .msg-group { display: flex; flex-direction: column; gap: 2px; margin-bottom: 10px; }
    .msg-group.user  { align-items: flex-end; }
    .msg-group.jerry { align-items: flex-start; }
    .msg-row { display: flex; align-items: flex-end; gap: 8px; max-width: 85%; }
    .msg-group.user  .msg-row { flex-direction: row-reverse; }
    .msg-group.jerry .msg-row { flex-direction: row; }
    .msg-avatar {
      width: 28px; height: 28px; border-radius: 50%;
      background: linear-gradient(135deg,#0a84ff,#30d158);
      display: flex; align-items: center; justify-content: center;
      font-size: 12px; font-weight: 700; color: #fff;
      flex-shrink: 0; align-self: flex-end;
    }
    .msg-group.user .msg-avatar { display: none; }
    .bubble {
      border-radius: 20px; padding: 10px 14px; font-size: 15px;
      line-height: 1.5; word-break: break-word; max-width: 100%;
    }
    .bubble.user { background: #0a84ff; color: #fff; border-bottom-right-radius: 5px; }
    .bubble.jerry { background: #1c1c1e; color: #f2f2f7; border-bottom-left-radius: 5px; }

    /* Markdown */
    .md p { margin: 0 0 8px; }
    .md p:last-child { margin-bottom: 0; }
    .md ul { margin: 6px 0 8px 18px; }
    .md li { margin: 3px 0; }
    .md strong { font-weight: 700; }
    .md code {
      background: rgba(255,255,255,0.12); border-radius: 6px;
      padding: 1px 5px; font-family: ui-monospace,SFMono-Regular,Menlo,monospace; font-size: 13px;
    }

    /* Code blocks */
    .code-block {
      background: #0d0d0f; border: 1px solid rgba(255,255,255,0.1);
      border-radius: 12px; margin: 8px 0; overflow: hidden;
      font-family: ui-monospace,SFMono-Regular,Menlo,monospace;
    }
    .code-header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 6px 12px; background: rgba(255,255,255,0.05);
      border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .code-lang { font-size: 11px; color: #636366; text-transform: uppercase; letter-spacing: 0.05em; }
    .code-copy-btn {
      background: none; border: none; color: #0a84ff; font-size: 11px;
      cursor: pointer; padding: 2px 6px; border-radius: 6px; touch-action: manipulation;
    }
    .code-copy-btn:active { background: rgba(10,132,255,0.15); }
    .code-block pre { margin: 0; padding: 12px; overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .code-block code { font-size: 13px; line-height: 1.5; color: #e5e5ea; white-space: pre; }

    /* Thinking details */
    .thinking-details {
      margin-bottom: 8px; background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; overflow: hidden;
    }
    .thinking-details summary {
      padding: 6px 10px; font-size: 11px; color: #636366;
      cursor: pointer; list-style: none; user-select: none;
    }
    .thinking-details summary::-webkit-details-marker { display: none; }
    .thinking-body {
      padding: 8px 10px; font-size: 12px; color: #8e8e93;
      line-height: 1.5; border-top: 1px solid rgba(255,255,255,0.06);
      white-space: pre-wrap; word-break: break-word;
    }

    /* Tool details */
    .tool-details {
      margin-bottom: 8px; background: rgba(48,209,88,0.06);
      border: 1px solid rgba(48,209,88,0.15); border-radius: 10px; overflow: hidden;
    }
    .tool-details summary {
      padding: 6px 10px; font-size: 11px; color: #30d158;
      cursor: pointer; list-style: none; user-select: none;
    }
    .tool-details summary::-webkit-details-marker { display: none; }
    .tool-event { padding: 8px 10px; border-top: 1px solid rgba(48,209,88,0.1); }
    .tool-name { font-size: 11px; color: #30d158; font-weight: 600; margin-bottom: 4px; }
    .tool-code, .tool-result {
      font-family: ui-monospace,monospace; font-size: 11px;
      background: rgba(0,0,0,0.3); border-radius: 6px;
      padding: 4px 8px; margin: 4px 0 0; color: #e5e5ea;
      overflow-x: auto; white-space: pre-wrap; word-break: break-all;
      -webkit-overflow-scrolling: touch;
    }
    .tool-result { color: #8e8e93; }

    /* Approval card */
    .approval-card {
      margin: 8px 16px;
      background: rgba(255,159,10,0.08);
      border: 1px solid rgba(255,159,10,0.3);
      border-radius: 14px;
      padding: 14px 16px;
      font-size: 13px;
    }
    .approval-card.approved {
      background: rgba(48,209,88,0.08);
      border-color: rgba(48,209,88,0.3);
    }
    .approval-card.denied {
      background: rgba(255,69,58,0.08);
      border-color: rgba(255,69,58,0.3);
    }
    .approval-header {
      font-weight: 700; font-size: 13px; color: #ff9f0a; margin-bottom: 8px;
    }
    .approval-card.approved .approval-header { color: #30d158; }
    .approval-card.denied .approval-header { color: #ff453a; }
    .approval-label { font-size: 11px; color: #636366; margin-bottom: 4px; }
    .approval-cmd {
      background: rgba(0,0,0,0.4); border-radius: 8px; padding: 8px 10px;
      font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px;
      color: #e5e5ea; white-space: pre-wrap; word-break: break-all;
      margin: 0 0 8px;
    }
    .approval-reason {
      font-size: 12px; color: #8e8e93; margin-bottom: 10px; line-height: 1.5;
    }
    .approval-actions { display: flex; gap: 8px; align-items: center; }
    .approval-btn {
      border-radius: 20px; border: none; font-size: 13px; font-weight: 600;
      padding: 7px 18px; cursor: pointer; touch-action: manipulation;
    }
    .approve-btn { background: #30d158; color: #000; }
    .deny-btn { background: rgba(255,255,255,0.1); color: #ff453a; }
    .approval-status { font-size: 12px; font-weight: 600; }
    .approval-status.approved { color: #30d158; }
    .approval-status.denied { color: #ff453a; }
    .approval-output {
      margin-top: 10px;
      background: rgba(0,0,0,0.4); border-radius: 8px; padding: 8px 10px;
      font-family: 'SF Mono', 'Fira Code', monospace; font-size: 11px;
      color: #8e8e93; white-space: pre-wrap; word-break: break-all;
      max-height: 200px; overflow-y: auto;
    }

    /* Attachment chips */
    .att-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
    .att-chip {
      display: inline-flex; align-items: center; gap: 5px;
      background: rgba(255,255,255,0.1); border-radius: 999px;
      padding: 4px 10px; font-size: 12px; color: #aeaeb2;
    }

    /* Meta */
    .msg-meta { font-size: 11px; color: #636366; padding: 0 6px; margin-top: 2px; }
    .msg-meta .copy-btn {
      background: none; border: none; color: #0a84ff; font-size: 11px;
      cursor: pointer; padding: 0 0 0 8px;
      touch-action: manipulation; -webkit-tap-highlight-color: transparent;
    }
    .latency-tag { color: #636366; }

    /* Typing indicator */
    .typing-row { display: flex; align-items: flex-end; gap: 8px; margin-bottom: 10px; }
    .typing-avatar {
      width: 28px; height: 28px; border-radius: 50%;
      background: linear-gradient(135deg,#0a84ff,#30d158);
      display: flex; align-items: center; justify-content: center;
      font-size: 12px; font-weight: 700; color: #fff; flex-shrink: 0;
    }
    .typing-bubble {
      background: #1c1c1e; border-radius: 20px; border-bottom-left-radius: 5px;
      padding: 14px 18px; display: inline-flex; align-items: center; gap: 6px;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: #636366; animation: imsg-bounce 1.3s infinite ease-in-out; }
    .dot:nth-child(1) { animation-delay: 0s; }
    .dot:nth-child(2) { animation-delay: 0.18s; }
    .dot:nth-child(3) { animation-delay: 0.36s; }
    @keyframes imsg-bounce {
      0%,60%,100% { transform: translateY(0); opacity: 0.4; }
      30%          { transform: translateY(-5px); opacity: 1; }
    }

    /* Composer */
    .composer {
      flex-shrink: 0; background: rgba(18,18,20,0.97);
      backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
      border-top: 1px solid rgba(255,255,255,0.08);
      padding: 8px 12px;
      padding-bottom: max(12px,env(safe-area-inset-bottom,12px));
      padding-left: max(12px,env(safe-area-inset-left));
      padding-right: max(12px,env(safe-area-inset-right));
    }
    /* Composer toolbar (replaces upload-bar) */
    .composer-toolbar {
      display: flex; gap: 6px; align-items: center; margin-bottom: 8px;
      flex-wrap: nowrap; overflow-x: auto; scrollbar-width: none;
    }
    .composer-toolbar::-webkit-scrollbar { display: none; }
    .upload-label {
      display: inline-flex; align-items: center; justify-content: center;
      background: rgba(255,255,255,0.08); border: none; border-radius: 20px;
      color: #8e8e93; font-size: 16px;
      width: 36px; height: 36px;
      cursor: pointer; flex-shrink: 0;
      touch-action: manipulation; -webkit-tap-highlight-color: transparent;
      transition: background 0.15s;
    }
    .upload-label:active { background: rgba(255,255,255,0.16); }
    .quick-toggle {
      display: inline-flex; align-items: center; gap: 4px;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 20px; color: #8e8e93;
      font-size: 12px; font-weight: 600;
      padding: 0 12px; height: 36px; white-space: nowrap;
      cursor: pointer; flex-shrink: 0;
      touch-action: manipulation; -webkit-tap-highlight-color: transparent;
      transition: all 0.15s;
    }
    .quick-toggle.active {
      background: rgba(48,209,88,0.18);
      border-color: rgba(48,209,88,0.4);
      color: #30d158;
    }
    .quick-toggle:active { opacity: 0.75; }
    .upload-hint { display: none; } /* moved hint into title attr */
    .upload-bar {
      display: flex; gap: 8px; align-items: center; margin-bottom: 8px;
      flex-wrap: nowrap; overflow-x: auto; -webkit-overflow-scrolling: touch;
    }
    .upload-label {
      display: inline-flex; align-items: center; gap: 5px;
      background: rgba(255,255,255,0.1); border: none; border-radius: 20px;
      color: #0a84ff; font-size: 13px; font-weight: 600;
      padding: 0 14px; height: 34px; cursor: pointer; flex-shrink: 0;
      touch-action: manipulation; -webkit-tap-highlight-color: transparent; transition: background 0.15s;
    }
    .upload-label:active { background: rgba(255,255,255,0.2); }
    .upload-hint { font-size: 11px; color: #636366; white-space: nowrap; }
    .file-preview-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
    .file-chip { display: inline-flex; align-items: center; gap: 6px; background: rgba(255,255,255,0.1); border-radius: 12px; padding: 5px 10px; font-size: 12px; }
    .file-chip button { background: none; border: none; color: #8e8e93; font-size: 18px; cursor: pointer; line-height: 1; padding: 2px; touch-action: manipulation; -webkit-tap-highlight-color: transparent; }
    .compose-row { display: flex; align-items: flex-end; gap: 10px; }
    .compose-input {
      flex: 1; background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.12); border-radius: 22px;
      color: #f2f2f7; font-size: 16px; padding: 10px 16px;
      resize: none; outline: none; max-height: 120px;
      overflow-y: auto; -webkit-overflow-scrolling: touch;
      line-height: 1.4; font-family: inherit; -webkit-appearance: none;
    }
    .compose-input::placeholder { color: #48484a; }
    .send-btn {
      width: 44px; height: 44px; border-radius: 50%;
      background: #0a84ff; border: none; color: #fff; font-size: 20px;
      display: flex; align-items: center; justify-content: center;
      cursor: pointer; flex-shrink: 0;
      touch-action: manipulation; -webkit-tap-highlight-color: transparent;
      user-select: none; -webkit-user-select: none; transition: background 0.15s, transform 0.1s;
    }
    .send-btn:disabled { background: #2c2c2e; cursor: default; }
    .send-btn.stop { background: #ff3b30; }
    .send-btn:active:not(:disabled) { transform: scale(0.92); }

    .error-bar { font-size: 12px; color: #ff453a; padding: 4px 4px 0; min-height: 0; }

    /* Chips row */
    .chips-row {
      flex-shrink: 0; display: flex; gap: 6px; padding: 6px 12px;
      padding-left: max(12px,env(safe-area-inset-left));
      padding-right: max(12px,env(safe-area-inset-right));
      background: rgba(18,18,20,0.97);
      border-bottom: 1px solid rgba(255,255,255,0.06);
      overflow-x: auto; -webkit-overflow-scrolling: touch; scrollbar-width: none;
    }
    .chips-row::-webkit-scrollbar { display: none; }
    .chip {
      display: inline-flex; align-items: center;
      background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.1);
      border-radius: 999px; padding: 4px 10px;
      font-size: 11px; color: #8e8e93; white-space: nowrap; flex-shrink: 0;
    }

    .footer { display: none; }

    @media (min-width: 600px) {
      body { max-width: 700px; margin: 0 auto; }
      .footer { display: flex; justify-content: space-between; align-items: center;
        padding: 5px 16px 8px; font-size: 11px; color: #48484a;
        background: rgba(18,18,20,0.96); flex-shrink: 0; }
      .bubble { font-size: 15px; }
      .compose-input { font-size: 15px; }
    }
  </style>
</head>
<body>

  <div class="header">
    <div class="header-inner">
      <div class="header-avatar">J</div>
      <div class="header-info">
        <div class="header-name">Jerry</div>
        <div id="headerStatus" class="header-status">Checking&#8230;</div>
      </div>
      <button id="settingsBtn" class="header-btn">Settings</button>
    </div>
  </div>

  <div class="chips-row">
    <div id="memoryChip"  class="chip">Memory: 0</div>
    <div id="modeChip"    class="chip">Mode: general</div>
    <div id="sessionChip" class="chip">Identity: unavailable</div>
  </div>

  <div id="drawer" class="settings-drawer">
    <div class="s-label">Conversation</div>
    <div class="settings-row">
      <select id="conversationSelect" class="s-select"></select>
    </div>
    <div class="settings-row">
      <button id="newChatBtn"    class="s-btn s-grey">New chat</button>
      <button id="deleteChatBtn" class="s-btn s-red">Delete chat</button>
    </div>
    <div class="settings-row">
      <button id="exportMdBtn"  class="s-btn s-grey">Export markdown</button>
      <button id="exportTxtBtn" class="s-btn s-grey">Export text</button>
    </div>
    <div class="s-label">Mode</div>
    <div class="settings-row">
      <select id="modeSelect" class="s-select">
        <option value="general">General</option>
        <option value="homelab">Homelab</option>
        <option value="devops">DevOps</option>
        <option value="coding">Coding</option>
      </select>
    </div>
    <div class="s-label">Options</div>
    <label class="s-toggle-row">
      <span>Deep thinking</span>
      <input type="checkbox" id="thinkToggle" />
    </label>
    <label class="s-toggle-row">
      <span>Use tools (kubectl, health checks, memory)</span>
      <input type="checkbox" id="toolsToggle" />
    </label>
    <div class="s-label">Profile memory</div>
    <div class="settings-row">
      <input id="preferredName" class="s-input" type="text" placeholder="Your preferred name" />
    </div>
    <div class="settings-row">
      <button id="saveProfileBtn"  class="s-btn s-blue">Save name</button>
      <button id="resetProfileBtn" class="s-btn s-grey">Reset name</button>
    </div>
    <div class="s-label">Search conversations</div>
    <div class="settings-row">
      <input id="searchInput" class="s-input" type="search" placeholder="Search conversations..." />
    </div>
    <div id="searchResults" class="search-results"></div>
    <div class="s-label">Persistent memory</div>
    <div id="memList" style="display:flex;flex-direction:column;gap:6px;"></div>
    <div style="display:flex;flex-direction:column;gap:6px;margin-top:4px;">
      <select id="memCatSelect" class="s-select" style="flex:none;">
        <option value="general">general</option>
        <option value="infra">infra</option>
        <option value="service">service</option>
        <option value="runbook">runbook</option>
        <option value="decision">decision</option>
      </select>
      <input id="memTitleInput" class="s-input" type="text" placeholder="Memory title" />
      <textarea id="memContentInput" class="s-input" rows="2" placeholder="Memory content" style="resize:none;height:auto;"></textarea>
      <button id="addMemBtn" class="s-btn s-green">Add memory</button>
    </div>
    <div class="settings-row">
      <button id="diagnosticsBtn" class="s-btn s-grey">Toggle diagnostics</button>
    </div>
    <div id="diagnosticsPanel" style="display:none;">
      <div id="diagnosticsText" class="s-diag"></div>
    </div>
  </div>

  <div id="chatArea" class="chat-area"></div>

  <div class="composer">
    <div class="composer-toolbar">
      <label class="upload-label">
        &#128206;
        <input id="fileInput" type="file" accept="image/png,image/jpeg,image/webp,application/pdf" multiple style="display:none;" />
      </label>
      <button id="toolsQuickBtn" class="quick-toggle" title="Enable tools so Jerry can run kubectl, curl, df, etc. directly">&#9881; Tools</button>
      <button id="thinkQuickBtn" class="quick-toggle" title="Enable deep thinking for complex problems">&#129504; Think</button>
      <button id="clearDraftBtn" class="quick-toggle" style="margin-left:auto;color:#636366;" title="Clear draft">&#10005;</button>
    </div>
    <div id="filePreviewRow" class="file-preview-row"></div>
    <div class="compose-row">
      <textarea id="prompt" class="compose-input" rows="1" placeholder="Message Jerry&#8230;"></textarea>
      <button id="sendBtn" class="send-btn" title="Send">&#x2191;</button>
    </div>
    <div id="errorBar" class="error-bar"></div>
  </div>

  <div class="footer">
    <span id="footerLeft">No responses yet.</span>
    <span id="footerRight">Local</span>
  </div>

<script>
  /* Element refs */
  const chatAreaEl        = document.getElementById('chatArea');
  const headerStatusEl    = document.getElementById('headerStatus');
  const footerLeftEl      = document.getElementById('footerLeft');
  const footerRightEl     = document.getElementById('footerRight');
  const promptEl          = document.getElementById('prompt');
  const sendBtnEl         = document.getElementById('sendBtn');
  const clearDraftBtnEl   = document.getElementById('clearDraftBtn');
  const errorBarEl        = document.getElementById('errorBar');
  const preferredNameEl   = document.getElementById('preferredName');
  const saveProfileBtnEl  = document.getElementById('saveProfileBtn');
  const resetProfileBtnEl = document.getElementById('resetProfileBtn');
  const memoryChipEl      = document.getElementById('memoryChip');
  const modeChipEl        = document.getElementById('modeChip');
  const sessionChipEl     = document.getElementById('sessionChip');
  const modeSelectEl      = document.getElementById('modeSelect');
  const convSelectEl      = document.getElementById('conversationSelect');
  const newChatBtnEl      = document.getElementById('newChatBtn');
  const deleteChatBtnEl   = document.getElementById('deleteChatBtn');
  const exportMdBtnEl     = document.getElementById('exportMdBtn');
  const exportTxtBtnEl    = document.getElementById('exportTxtBtn');
  const diagBtnEl         = document.getElementById('diagnosticsBtn');
  const diagPanelEl       = document.getElementById('diagnosticsPanel');
  const diagTextEl        = document.getElementById('diagnosticsText');
  const fileInputEl       = document.getElementById('fileInput');
  const filePreviewRowEl  = document.getElementById('filePreviewRow');
  const drawerEl          = document.getElementById('drawer');
  const settingsBtnEl     = document.getElementById('settingsBtn');
  const thinkToggleEl     = document.getElementById('thinkToggle');
  const toolsToggleEl     = document.getElementById('toolsToggle');
  const toolsQuickBtnEl   = document.getElementById('toolsQuickBtn');
  const thinkQuickBtnEl   = document.getElementById('thinkQuickBtn');
  const searchInputEl     = document.getElementById('searchInput');
  const searchResultsEl   = document.getElementById('searchResults');
  const memListEl         = document.getElementById('memList');
  const memCatSelectEl    = document.getElementById('memCatSelect');
  const memTitleInputEl   = document.getElementById('memTitleInput');
  const memContentInputEl = document.getElementById('memContentInput');
  const addMemBtnEl       = document.getElementById('addMemBtn');

  /* State */
  let conversations       = [];
  let activeConvId        = null;
  let activeConv          = null;
  let selectedAttachments = [];
  let streamController    = null;
  let streaming           = false;
  let drawerOpen          = false;
  let diagOpen            = false;
  let typingVisible       = false;

  /* Helpers */
  function esc(str) {
    return String(str)
      .replaceAll('&','&amp;').replaceAll('<','&lt;')
      .replaceAll('>','&gt;').replaceAll('"','&quot;');
  }

  function renderInlineMd(text) {
    let h = esc(text);
    h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
    h = h.replace(/[*][*]([^*]+)[*][*]/g, '<strong>$1</strong>');
    return h;
  }

  function renderMd(text) {
    const CODE_FENCE = /```(\\w*)\\n?([\\s\\S]*?)```/g;
    const blocks = [];
    const placeholder_text = String(text || '').replace(CODE_FENCE, (_, lang, code) => {
      const idx = blocks.length;
      blocks.push({ lang: lang || 'text', code: code.trimEnd() });
      return '\\x00CODEBLOCK' + idx + '\\x00';
    });

    const lines = placeholder_text.split('\\n');
    let html = '', inList = false, para = [];

    function flushPara() {
      if (!para.length) return;
      const joined = para.join(' ');
      if (joined.startsWith('\\x00CODEBLOCK') && joined.endsWith('\\x00')) {
        const idx = parseInt(joined.replace(/\\x00CODEBLOCK(\\d+)\\x00/, '$1'));
        const b = blocks[idx];
        if (b) {
          const safeCode = b.code.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
          html += '<div class="code-block"><div class="code-header"><span class="code-lang">' + esc(b.lang) + '</span><button class="code-copy-btn" data-code="' + b.code.replace(/"/g,'&quot;') + '">Copy</button></div><pre><code>' + safeCode + '</code></pre></div>';
        }
      } else {
        html += '<p>' + renderInlineMd(joined) + '</p>';
      }
      para = [];
    }

    function closeList() { if (inList) { html += '</ul>'; inList = false; } }

    for (const raw of lines) {
      const line = raw.trimEnd();
      if (line.startsWith('\\x00CODEBLOCK') && line.endsWith('\\x00')) {
        flushPara(); closeList();
        const idx = parseInt(line.replace(/\\x00CODEBLOCK(\\d+)\\x00/, '$1'));
        const b = blocks[idx];
        if (b) {
          const safeCode = b.code.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
          html += '<div class="code-block"><div class="code-header"><span class="code-lang">' + esc(b.lang || 'text') + '</span><button class="code-copy-btn" data-code="' + b.code.replace(/"/g,'&quot;') + '">Copy</button></div><pre><code>' + safeCode + '</code></pre></div>';
        }
        continue;
      }
      if (!line.trim()) { flushPara(); closeList(); continue; }
      const li = line.match(/^[-*]\\s+(.*)$/);
      if (li) {
        flushPara();
        if (!inList) { html += '<ul>'; inList = true; }
        html += '<li>' + renderInlineMd(li[1]) + '</li>';
        continue;
      }
      closeList();
      para.push(line.trim());
    }
    flushPara(); closeList();
    return html || '<p></p>';
  }

  function fmtTime(iso) {
    try { return new Date(iso).toLocaleTimeString([], { hour:'numeric', minute:'2-digit' }); }
    catch { return ''; }
  }

  function setError(msg) { errorBarEl.textContent = msg || ''; }

  function updateChips(name, mode, memCount) {
    memoryChipEl.textContent = 'Memory: ' + (memCount !== undefined ? memCount : '?');
    modeChipEl.textContent   = 'Mode: ' + mode;
  }

  function setDrawer(open) {
    drawerOpen = open;
    drawerEl.classList.toggle('open', open);
    settingsBtnEl.textContent = open ? 'Close' : 'Settings';
    if (open) { loadMemories(); }
  }

  function setStreaming(on) {
    streaming = on;
    sendBtnEl.classList.toggle('stop', on);
    sendBtnEl.innerHTML = on ? '&#9632;' : '&#x2191;';
    sendBtnEl.title     = on ? 'Stop' : 'Send';
  }

  function showTyping() {
    if (typingVisible) return;
    typingVisible = true;
    const row = document.createElement('div');
    row.className = 'typing-row'; row.id = 'typingIndicator';
    row.innerHTML = '<div class="typing-avatar">J</div><div class="typing-bubble"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
    chatAreaEl.appendChild(row);
    chatAreaEl.scrollTop = chatAreaEl.scrollHeight;
  }

  function hideTyping() {
    typingVisible = false;
    const el = document.getElementById('typingIndicator');
    if (el) el.remove();
  }

  function renderConvSelect() {
    convSelectEl.innerHTML = '';
    if (!conversations.length) {
      const o = document.createElement('option');
      o.value = ''; o.textContent = 'No conversations yet';
      convSelectEl.appendChild(o); convSelectEl.disabled = true; return;
    }
    convSelectEl.disabled = false;
    for (const c of conversations) {
      const o = document.createElement('option');
      o.value = String(c.id); o.textContent = c.title;
      convSelectEl.appendChild(o);
    }
    if (activeConvId && conversations.some(c => c.id === activeConvId)) {
      convSelectEl.value = String(activeConvId);
    } else {
      convSelectEl.value = String(conversations[0].id);
    }
  }

  function renderUploads() {
    filePreviewRowEl.innerHTML = '';
    for (const item of selectedAttachments) {
      const chip = document.createElement('div');
      chip.className = 'file-chip';
      const span = document.createElement('span');
      span.textContent = item.name;
      const rm = document.createElement('button');
      rm.textContent = '\\xd7';
      rm.addEventListener('click', () => {
        selectedAttachments = selectedAttachments.filter(a => a.local_id !== item.local_id);
        renderUploads();
      });
      chip.appendChild(span); chip.appendChild(rm);
      filePreviewRowEl.appendChild(chip);
    }
  }

  async function copyText(text) {
    try { await navigator.clipboard.writeText(text); }
    catch { setError('Copy failed.'); }
  }

  function renderWelcome() {
    chatAreaEl.innerHTML = '<div class="welcome"><div class="w-avatar">J</div><h2>Hi, I\'m Jerry.</h2><p>I can help with your homelab, DevOps work, coding, screenshots, and PDF analysis. Enable tools in Settings to inspect live infra.<br><br>Upload a PNG, JPG, WebP, or PDF for analysis.</p></div>';
    footerLeftEl.textContent = 'No responses yet.';
    footerRightEl.textContent = 'Local';
  }

  function buildThinkingEl(thinkingText) {
    const det = document.createElement('details');
    det.className = 'thinking-details';
    const sum = document.createElement('summary');
    sum.textContent = "Jerry's reasoning";
    const body = document.createElement('div');
    body.className = 'thinking-body';
    body.textContent = thinkingText;
    det.appendChild(sum); det.appendChild(body);
    return det;
  }

  /* Approval card — rendered inline in chat area when Jerry proposes a write command */
  function renderApprovalCard(evt) {
    const card = document.createElement('div');
    card.className = 'approval-card';
    card.dataset.writeId = evt.id;
    card.innerHTML =
      '<div class="approval-header">\\u26a0\\ufe0f Approval Required</div>' +
      '<div class="approval-label">Command to execute:</div>' +
      '<pre class="approval-cmd">' + esc(evt.command || '') + '</pre>' +
      (evt.reasoning ? '<div class="approval-reason">' + esc(evt.reasoning) + '</div>' : '') +
      '<div class="approval-actions">' +
        '<button class="approval-btn approve-btn">Approve &amp; Run</button>' +
        '<button class="approval-btn deny-btn">Deny</button>' +
      '</div>' +
      '<div class="approval-output" style="display:none;"></div>';

    card.querySelector('.approve-btn').addEventListener('click', async () => {
      card.querySelector('.approval-actions').innerHTML = '<span style="color:#636366;font-size:12px;">Executing\\u2026</span>';
      try {
        const res = await fetch('/api/writes/' + evt.id + '/approve', { method: 'POST' });
        const d = await res.json();
        const out = card.querySelector('.approval-output');
        out.style.display = 'block';
        if (res.ok) {
          card.classList.add('approved');
          card.querySelector('.approval-actions').innerHTML = '<span class="approval-status approved">\\u2713 Approved &amp; Executed</span>';
          out.textContent = d.output || '(no output)';
        } else {
          card.classList.add('denied');
          card.querySelector('.approval-actions').innerHTML = '<span class="approval-status denied">\\u2717 Error: ' + esc(d.detail || 'unknown') + '</span>';
        }
      } catch (err) {
        card.querySelector('.approval-actions').innerHTML = '<span class="approval-status denied">\\u2717 Request failed</span>';
      }
    });

    card.querySelector('.deny-btn').addEventListener('click', async () => {
      try {
        await fetch('/api/writes/' + evt.id + '/deny', { method: 'POST' });
      } catch (_) {}
      card.classList.add('denied');
      card.querySelector('.approval-actions').innerHTML = '<span class="approval-status denied">\\u2717 Denied</span>';
    });

    chatAreaEl.appendChild(card);
    chatAreaEl.scrollTop = chatAreaEl.scrollHeight;
  }

  function buildToolEventsEl(toolEvents) {
    const det = document.createElement('details');
    det.className = 'tool-details';
    const sum = document.createElement('summary');
    sum.textContent = '\\ud83d\\udd27 Used tools (' + toolEvents.length + ')';
    det.appendChild(sum);
    for (const ev of toolEvents) {
      const div = document.createElement('div');
      div.className = 'tool-event';
      const nameEl = document.createElement('div');
      nameEl.className = 'tool-name'; nameEl.textContent = ev.tool || ev.name || '';
      div.appendChild(nameEl);
      if (ev.args) {
        const argsEl = document.createElement('pre');
        argsEl.className = 'tool-code';
        const cmd = ev.args.command || ev.args.url || JSON.stringify(ev.args);
        argsEl.textContent = cmd;
        div.appendChild(argsEl);
      }
      if (ev.result != null) {
        const resEl = document.createElement('pre');
        resEl.className = 'tool-result';
        resEl.textContent = String(ev.result).slice(0, 1000);
        div.appendChild(resEl);
      }
      det.appendChild(div);
    }
    return det;
  }

  function renderMessages() {
    if (!activeConv || !activeConv.messages || !activeConv.messages.length) {
      renderWelcome(); return;
    }
    chatAreaEl.innerHTML = '';
    let lastLatency = null, lastModel = null;
    const groups = [];
    for (const msg of activeConv.messages) {
      if (!groups.length || groups[groups.length-1].role !== msg.role) {
        groups.push({ role: msg.role, msgs: [msg] });
      } else {
        groups[groups.length-1].msgs.push(msg);
      }
    }

    for (const group of groups) {
      const groupEl = document.createElement('div');
      groupEl.className = 'msg-group ' + (group.role === 'user' ? 'user' : 'jerry');

      for (const msg of group.msgs) {
        const rowEl = document.createElement('div');
        rowEl.className = 'msg-row';

        if (group.role !== 'user') {
          const av = document.createElement('div');
          av.className = 'msg-avatar'; av.textContent = 'J';
          rowEl.appendChild(av);
        }

        const bubbleWrapper = document.createElement('div');
        bubbleWrapper.style.display = 'flex';
        bubbleWrapper.style.flexDirection = 'column';
        bubbleWrapper.style.maxWidth = '100%';

        // Thinking block
        if (msg.thinking && msg.thinking.trim()) {
          bubbleWrapper.appendChild(buildThinkingEl(msg.thinking));
        }

        // Tool events block
        const toolEvs = msg.tool_events || [];
        if (toolEvs.length) {
          bubbleWrapper.appendChild(buildToolEventsEl(toolEvs));
        }

        const bubbleEl = document.createElement('div');
        bubbleEl.className = 'bubble ' + (group.role === 'user' ? 'user' : 'jerry');

        if (group.role === 'user') {
          bubbleEl.textContent = msg.content;
        } else {
          const md = document.createElement('div');
          md.className = 'md';
          md.innerHTML = renderMd(msg.content);
          // Wire copy buttons
          for (const btn of md.querySelectorAll('.code-copy-btn')) {
            btn.addEventListener('click', () => {
              navigator.clipboard.writeText(btn.dataset.code).then(() => {
                btn.textContent = 'Copied!';
                setTimeout(() => btn.textContent = 'Copy', 1500);
              });
            });
          }
          bubbleEl.appendChild(md);
        }

        bubbleWrapper.appendChild(bubbleEl);
        rowEl.appendChild(bubbleWrapper);
        groupEl.appendChild(rowEl);

        if (msg.attachments && msg.attachments.length) {
          const attRow = document.createElement('div');
          attRow.className = 'att-row';
          for (const att of msg.attachments) {
            const chip = document.createElement('div');
            chip.className = 'att-chip';
            chip.textContent = att.kind.toUpperCase() + ': ' + att.name;
            attRow.appendChild(chip);
          }
          groupEl.appendChild(attRow);
        }

        const metaEl = document.createElement('div');
        metaEl.className = 'msg-meta';
        let metaHTML = '<span>' + fmtTime(msg.created_at) + '</span>';
        if (group.role !== 'user') {
          const safeContent = msg.content.replace(/"/g, '&quot;');
          metaHTML += '<button class="copy-btn" data-content="' + safeContent + '">Copy</button>';
          if (msg.latency_ms != null) {
            metaHTML += '<span class="latency-tag"> &middot; ' + (msg.latency_ms/1000).toFixed(2) + 's</span>';
            lastLatency = msg.latency_ms;
          }
          if (msg.model_used) {
            metaHTML += '<span class="latency-tag"> &middot; ' + esc(msg.model_used) + '</span>';
            lastModel = msg.model_used;
          }
        }
        metaEl.innerHTML = metaHTML;
        for (const btn of metaEl.querySelectorAll('.copy-btn')) {
          btn.addEventListener('click', () => copyText(btn.dataset.content));
        }
        groupEl.appendChild(metaEl);
      }
      chatAreaEl.appendChild(groupEl);
    }

    if (lastLatency !== null) {
      footerLeftEl.textContent = 'Last: ' + (lastLatency/1000).toFixed(2) + 's';
      footerRightEl.textContent = lastModel || 'Local';
    } else {
      footerLeftEl.textContent = 'Conversation loaded.';
      footerRightEl.textContent = lastModel || 'Local';
    }
    chatAreaEl.scrollTop = chatAreaEl.scrollHeight;
  }

  /* Memory panel */
  async function loadMemories() {
    const res = await fetch('/api/memory');
    const d = await res.json();
    const mems = d.memories || [];
    renderMemories(mems);
    memoryChipEl.textContent = 'Memory: ' + mems.length;
  }

  function renderMemories(mems) {
    memListEl.innerHTML = '';
    if (!mems.length) {
      memListEl.innerHTML = '<div style="font-size:12px;color:#636366;">No memories yet. Jerry will save things when you tell him to, or when tools call save_memory.</div>';
      return;
    }
    for (const m of mems) {
      const el = document.createElement('div');
      el.className = 'memory-item';
      el.innerHTML = '<div class="memory-item-body"><div class="memory-cat">' + esc(m.category) + '</div><div class="memory-title">' + esc(m.title) + '</div><div class="memory-content">' + esc(m.content) + '</div></div><button class="memory-del" data-id="' + m.id + '">\xd7</button>';
      el.querySelector('.memory-del').addEventListener('click', async () => {
        await fetch('/api/memory/' + m.id, { method: 'DELETE' });
        await loadMemories();
      });
      memListEl.appendChild(el);
    }
  }

  addMemBtnEl.addEventListener('click', async () => {
    const title   = memTitleInputEl.value.trim();
    const content = memContentInputEl.value.trim();
    const cat     = memCatSelectEl.value;
    if (!title || !content) { setError('Memory title and content required.'); return; }
    setError('');
    const res = await fetch('/api/memory', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ category: cat, title, content })
    });
    if (!res.ok) { setError('Failed to add memory.'); return; }
    memTitleInputEl.value = '';
    memContentInputEl.value = '';
    await loadMemories();
  });

  /* Search */
  let searchTimer = null;
  searchInputEl.addEventListener('input', () => {
    clearTimeout(searchTimer);
    const q = searchInputEl.value.trim();
    if (!q) { searchResultsEl.innerHTML = ''; return; }
    searchTimer = setTimeout(async () => {
      const res = await fetch('/api/search?q=' + encodeURIComponent(q) + '&limit=10');
      const d = await res.json();
      renderSearchResults(d.results || []);
    }, 300);
  });

  function renderSearchResults(results) {
    searchResultsEl.innerHTML = '';
    for (const r of results) {
      const el = document.createElement('div');
      el.className = 'search-result';
      const snippet = r.content.slice(0, 120) + (r.content.length > 120 ? '...' : '');
      el.innerHTML = '<div class="search-result-conv">' + esc(r.conversation_title) + ' \xb7 ' + r.role + '</div><div class="search-result-snippet">' + esc(snippet) + '</div>';
      el.addEventListener('click', async () => {
        setDrawer(false);
        await loadConversation(r.conversation_id);
      });
      searchResultsEl.appendChild(el);
    }
    if (!results.length) searchResultsEl.innerHTML = '<div style="font-size:12px;color:#636366;padding:4px 0;">No results.</div>';
  }

  /* API */
  async function checkHealth() {
    const ctrl = new AbortController();
    const tid = setTimeout(() => ctrl.abort(), 5000);
    try {
      const res = await fetch('/health', { signal: ctrl.signal });
      clearTimeout(tid);
      const d = await res.json();
      const name = d.preferred_name ? ' \xb7 ' + d.preferred_name : '';
      headerStatusEl.textContent = 'Online \xb7 ' + d.text_model + name;
      headerStatusEl.style.color = '';
      memoryChipEl.textContent = 'Memory: ' + (d.memory_count || 0);
    } catch {
      clearTimeout(tid);
      headerStatusEl.textContent = 'Backend unreachable';
      headerStatusEl.style.color = '#ff453a';
    }
  }

  async function loadSession() {
    try {
      const res = await fetch('/api/session');
      const d = await res.json();
      sessionChipEl.textContent = d.identity ? 'Identity: ' + d.identity : 'Identity: unavailable';
    } catch {
      sessionChipEl.textContent = 'Identity: unavailable';
    }
  }

  async function loadDiagnostics() {
    try {
      const res = await fetch('/api/diagnostics');
      const d = await res.json();
      diagTextEl.textContent = JSON.stringify(d, null, 2);
    } catch {
      diagTextEl.textContent = 'Failed to load diagnostics.';
    }
  }

  async function loadProfile() {
    const res = await fetch('/api/profile');
    const d = await res.json();
    preferredNameEl.value = d.preferred_name || '';
    updateChips(d.preferred_name || '', modeSelectEl.value);
  }

  async function saveProfile() {
    setError('');
    const res = await fetch('/api/profile', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preferred_name: preferredNameEl.value })
    });
    if (!res.ok) { setError(await res.text() || 'Save failed.'); return; }
    const d = await res.json();
    preferredNameEl.value = d.preferred_name || '';
    await checkHealth();
  }

  async function resetProfile() {
    setError('');
    const res = await fetch('/api/profile/reset', { method: 'POST' });
    if (!res.ok) { setError('Reset failed.'); return; }
    preferredNameEl.value = '';
    await checkHealth();
  }

  async function refreshConversations() {
    const res = await fetch('/api/conversations');
    const d = await res.json();
    conversations = d.conversations || [];
    renderConvSelect();
  }

  async function loadConversation(id) {
    if (!id) { activeConvId = null; activeConv = null; renderMessages(); return; }
    const res = await fetch('/api/conversations/' + id);
    if (!res.ok) { setError('Failed to load conversation.'); return; }
    const d = await res.json();
    activeConv   = d.conversation;
    activeConvId = activeConv.id;
    modeSelectEl.value = activeConv.mode || 'general';
    updateChips(preferredNameEl.value || '', modeSelectEl.value);
    renderMessages();
  }

  async function createNewChat() {
    setError('');
    const res = await fetch('/api/conversations', { method: 'POST' });
    if (!res.ok) { setError('Failed to create chat.'); return; }
    const d = await res.json();
    await refreshConversations();
    await loadConversation(d.conversation_id);
  }

  async function deleteCurrentChat() {
    if (!activeConvId) return;
    if (!window.confirm('Delete this conversation?')) return;
    const res = await fetch('/api/conversations/' + activeConvId, { method: 'DELETE' });
    if (!res.ok) { setError('Delete failed.'); return; }
    activeConvId = null; activeConv = null;
    await refreshConversations();
    if (conversations.length) { await loadConversation(conversations[0].id); }
    else { renderMessages(); }
  }

  async function fileToBase64(file) {
    const buf = await file.arrayBuffer();
    let bin = '';
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < bytes.length; i += 0x8000) {
      bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    }
    return btoa(bin);
  }

  async function handleFiles(fileList) {
    setError('');
    for (const file of Array.from(fileList || [])) {
      const mime = file.type || '';
      if (!['image/png','image/jpeg','image/webp','application/pdf'].includes(mime)) {
        setError('Unsupported: ' + file.name + '. Use PNG, JPG, WebP, or PDF.'); continue;
      }
      const data_base64 = await fileToBase64(file);
      selectedAttachments.push({ local_id: crypto.randomUUID(), name: file.name, mime_type: mime, data_base64 });
    }
    renderUploads();
  }

  function ensureActiveConv() {
    if (!activeConv) {
      activeConv = { id: activeConvId, title: 'New chat', mode: modeSelectEl.value, messages: [] };
    }
  }

  function pushOptimisticUser(text, atts) {
    ensureActiveConv();
    activeConv.messages.push({
      id: 'tmp-u-' + Date.now(), role: 'user',
      content: text || '[Attachment]',
      created_at: new Date().toISOString(),
      latency_ms: null,
      attachments: atts.map(a => ({ name: a.name, mime_type: a.mime_type, kind: a.mime_type === 'application/pdf' ? 'pdf' : 'image' })),
      model_used: null, tool_events: []
    });
    renderMessages();
  }

  function pushStreamingPlaceholder() {
    ensureActiveConv();
    const msg = {
      id: 'tmp-a-' + Date.now(), role: 'assistant',
      content: '', thinking: '', tool_events: [],
      created_at: new Date().toISOString(),
      latency_ms: null, attachments: [], model_used: null
    };
    activeConv.messages.push(msg);
    renderMessages();
    return msg;
  }

  function appendToken(placeholder, token) {
    placeholder.content += token;
    renderMessages();
    chatAreaEl.scrollTop = chatAreaEl.scrollHeight;
  }

  function autoResize() {
    promptEl.style.height = 'auto';
    promptEl.style.height = Math.min(promptEl.scrollHeight, 120) + 'px';
  }

  async function streamMessage() {
    if (streaming) {
      if (streamController) streamController.abort();
      return;
    }

    const text = promptEl.value.trim();
    if (!text && !selectedAttachments.length) return;

    setError('');
    if (!activeConvId) await createNewChat();

    const pendingText = text;
    const pendingAtts = [...selectedAttachments];

    promptEl.value = '';
    autoResize();
    selectedAttachments = [];
    renderUploads();

    pushOptimisticUser(pendingText, pendingAtts);
    const placeholder = pushStreamingPlaceholder();

    setStreaming(true);
    showTyping();
    streamController = new AbortController();

    // State for streaming
    let toolEvents = [];
    let thinkingContent = '';

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation_id: activeConvId,
          mode: modeSelectEl.value,
          message: pendingText,
          attachments: pendingAtts.map(a => ({ name: a.name, mime_type: a.mime_type, data_base64: a.data_base64 })),
          think: thinkToggleEl.checked,
          tools_enabled: toolsToggleEl.checked
        }),
        signal: streamController.signal
      });

      if (!response.ok) {
        let detail = 'HTTP ' + response.status;
        try { const j = await response.json(); detail = j.detail || JSON.stringify(j); } catch {}
        throw new Error(detail);
      }

      if (!response.body || !response.body.getReader) {
        throw new Error('Streaming not supported in this browser.');
      }

      const reader  = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer    = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\\n\\n');
        buffer = parts.pop() || '';

        for (const part of parts) {
          if (!part.startsWith('data: ')) continue;
          let evt;
          try { evt = JSON.parse(part.slice(6)); } catch { continue; }

          if (evt.type === 'thinking') {
            thinkingContent += evt.token || '';
            placeholder.thinking = thinkingContent;
            renderMessages();
          } else if (evt.type === 'token') {
            hideTyping();
            appendToken(placeholder, evt.token || '');
          } else if (evt.type === 'tool_call') {
            showTyping();
            toolEvents.push({ tool: evt.name || '', args: evt.args || {}, result: null });
            placeholder.tool_events = toolEvents;
            renderMessages();
          } else if (evt.type === 'tool_result') {
            const last = toolEvents[toolEvents.length - 1];
            if (last) last.result = evt.result;
            placeholder.tool_events = [...toolEvents];
            renderMessages();
          } else if (evt.type === 'approval_request') {
            renderApprovalCard(evt);
          } else if (evt.type === 'done') {
            activeConvId = evt.conversation_id;
            activeConv   = evt.conversation;
            await refreshConversations();
            renderConvSelect();
            convSelectEl.value = String(activeConvId);
            renderMessages();
            hideTyping();
            footerLeftEl.textContent  = 'Last: ' + (evt.latency_ms/1000).toFixed(2) + 's';
            footerRightEl.textContent = evt.model_used || 'Local';
            await checkHealth();
            return;
          } else if (evt.type === 'error') {
            throw new Error(evt.detail || 'Stream error.');
          }
        }
      }

      if (activeConvId) await loadConversation(activeConvId);

    } catch (err) {
      if (err && err.name === 'AbortError') {
        setError('Stopped.');
      } else {
        setError(err && err.message ? err.message : String(err));
      }
      if (activeConvId) await loadConversation(activeConvId);
    } finally {
      hideTyping();
      setStreaming(false);
      streamController = null;
      promptEl.focus();
    }
  }

  /* Event listeners */
  convSelectEl.addEventListener('change', e => {
    loadConversation(e.target.value ? Number(e.target.value) : null);
  });
  modeSelectEl.addEventListener('change', () => {
    updateChips(preferredNameEl.value || '', modeSelectEl.value);
  });

  saveProfileBtnEl.addEventListener('click',  saveProfile);
  resetProfileBtnEl.addEventListener('click', resetProfile);
  newChatBtnEl.addEventListener('click',      createNewChat);
  deleteChatBtnEl.addEventListener('click',   deleteCurrentChat);
  sendBtnEl.addEventListener('click',         streamMessage);

  exportMdBtnEl.addEventListener('click', () => {
    if (activeConvId) window.open('/api/conversations/' + activeConvId + '/export.md', '_blank');
  });
  exportTxtBtnEl.addEventListener('click', () => {
    if (activeConvId) window.open('/api/conversations/' + activeConvId + '/export.txt', '_blank');
  });

  diagBtnEl.addEventListener('click', async () => {
    diagOpen = !diagOpen;
    diagPanelEl.style.display = diagOpen ? 'block' : 'none';
    if (diagOpen) await loadDiagnostics();
  });

  settingsBtnEl.addEventListener('click', () => setDrawer(!drawerOpen));

  /* Quick toolbar toggles — sync with settings checkboxes */
  function syncQuickToggles() {
    toolsQuickBtnEl.classList.toggle('active', toolsToggleEl.checked);
    thinkQuickBtnEl.classList.toggle('active', thinkToggleEl.checked);
  }

  toolsQuickBtnEl.addEventListener('click', () => {
    toolsToggleEl.checked = !toolsToggleEl.checked;
    syncQuickToggles();
  });

  thinkQuickBtnEl.addEventListener('click', () => {
    thinkToggleEl.checked = !thinkToggleEl.checked;
    syncQuickToggles();
  });

  /* Keep quick toggles in sync when changed via settings drawer */
  toolsToggleEl.addEventListener('change', syncQuickToggles);
  thinkToggleEl.addEventListener('change', syncQuickToggles);
  syncQuickToggles(); /* reflect initial state on boot */

  clearDraftBtnEl.addEventListener('click', () => {
    promptEl.value = '';
    autoResize();
    selectedAttachments = [];
    renderUploads();
    setError('');
    promptEl.focus();
  });

  fileInputEl.addEventListener('change', async e => {
    await handleFiles(e.target.files);
    fileInputEl.value = '';
  });

  promptEl.addEventListener('input', autoResize);
  promptEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      streamMessage();
    }
  });

  async function boot() {
    setDrawer(false);
    setStreaming(false);
    /* Run independent init tasks in parallel; don't let health hang block chat load */
    await Promise.allSettled([checkHealth(), loadProfile(), loadSession()]);
    await refreshConversations();
    if (conversations.length) {
      await loadConversation(conversations[0].id);
    } else {
      renderWelcome();
    }
  }

  boot();
</script>
</body>
</html>
"""
