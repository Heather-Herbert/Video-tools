"""
llm.py — the model backend, and the one place that reads secrets.

Chain: Claude CLI → agy CLI → DeepSeek → OpenRouter. The two CLI backends cost
nothing and run locally, so they are tried first; the paid APIs are fallbacks.
A stage asks for either JSON (`classify`) or prose (`prose`) and does not care
which backend answered.

Secrets come from `.env` in the repo root, or from the real environment, which
wins. Nothing in this module hard-codes a key or a path outside the repo.

If the surrounding workspace happens to have its own `llm_backend.py` (the
Jennifer assistant does), set LLM_BACKEND_PATH and this module delegates to it
instead, so there is one chain to maintain on that machine. The repo does not
depend on that file existing.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

AGY_MODEL = os.environ.get("AGY_MODEL", "gemini-3.5-flash-medium")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/auto")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEEPSEEK_BASE = "https://api.deepseek.com"

# A CLI backend that answers with agent chatter instead of doing the task has
# failed, even though it exited 0. Treat these as a miss and fall through.
_NARRATION_MARKERS = (
    "i am unable to",
    "i cannot access",
    "i don't have access",
    "let me know if",
    "i'll help you",
    "as an ai",
)


class LLMError(RuntimeError):
    """Every backend in the chain failed."""


# --- environment ------------------------------------------------------------

@lru_cache(maxsize=1)
def load_env() -> dict:
    """
    Read `.env` into os.environ without overwriting anything already set.

    Real environment variables win, so CI and systemd units can override the
    file. Returns the parsed mapping for callers that want to inspect it.
    """
    parsed: dict[str, str] = {}
    env_file = Path(os.environ.get("VIDEO_TOOLS_ENV", REPO_ROOT / ".env"))
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            parsed[key] = value
            os.environ.setdefault(key, value)
    return parsed


def env(name: str, default: str = "") -> str:
    load_env()
    return os.environ.get(name, default)


def _openai_client(api_key: str, base_url: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url)


def openrouter_client():
    """The final-fallback client, or None when no key is configured."""
    key = env("OPENROUTER_API_KEY")
    return _openai_client(key, OPENROUTER_BASE) if key else None


# --- optional delegation to a host workspace backend ------------------------

@lru_cache(maxsize=1)
def _host_backend():
    """Load the workspace's own llm_backend.py when LLM_BACKEND_PATH points at one."""
    target = env("LLM_BACKEND_PATH")
    if not target or not Path(target).exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("host_llm_backend", target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return None                       # a broken host backend is not fatal
    return module if hasattr(module, "_llm_classify") else None


# --- individual backends ----------------------------------------------------

def _run_cli(argv: list[str], prompt: str, timeout: int) -> str:
    out = subprocess.run(
        argv, input=prompt, capture_output=True, text=True,
        timeout=timeout, check=True,
    )
    return out.stdout.strip()


def call_claude_cli(prompt: str, timeout: int = 120) -> str:
    if not shutil.which("claude"):
        raise LLMError("claude CLI not installed")
    return _run_cli(["claude", "-p", "--permission-mode", "plan"], prompt, timeout)


def call_agy(prompt: str, timeout: int = 120) -> str:
    if not shutil.which("agy"):
        raise LLMError("agy CLI not installed")
    return _run_cli(["agy", "-m", AGY_MODEL], prompt, timeout)


def call_deepseek(prompt: str, timeout: int = 120, max_tokens: int = 4000) -> str:
    key = env("DEEPSEEK_API_KEY")
    if not key:
        raise LLMError("DEEPSEEK_API_KEY not set")
    client = _openai_client(key, DEEPSEEK_BASE)
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, timeout=timeout,
    )
    return (resp.choices[0].message.content or "").strip()


def call_openrouter(prompt: str, client=None, timeout: int = 120,
                    max_tokens: int = 4000) -> str:
    client = client or openrouter_client()
    if client is None:
        raise LLMError("OPENROUTER_API_KEY not set")
    resp = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, timeout=timeout,
    )
    return (resp.choices[0].message.content or "").strip()


def _looks_like_narration(text: str) -> bool:
    head = text[:400].lower()
    return any(marker in head for marker in _NARRATION_MARKERS)


def _chain(prompt: str, client, max_tokens: int) -> str:
    """Walk the backend chain, returning the first usable answer."""
    attempts = [
        ("Claude CLI", lambda: call_claude_cli(prompt)),
        ("agy CLI", lambda: call_agy(prompt)),
        ("DeepSeek", lambda: call_deepseek(prompt, max_tokens=max_tokens)),
        ("OpenRouter", lambda: call_openrouter(prompt, client, max_tokens=max_tokens)),
    ]
    failures = []
    for name, call in attempts:
        try:
            answer = call()
        except Exception as exc:                       # noqa: BLE001 — try the next one
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if not answer:
            failures.append(f"{name}: empty response")
        elif _looks_like_narration(answer):
            failures.append(f"{name}: returned agent narration, not an answer")
        else:
            return answer
    raise LLMError("every LLM backend failed:\n  " + "\n  ".join(failures))


# --- public interface -------------------------------------------------------

def prose(prompt: str, client=None, max_tokens: int = 2000) -> str:
    """Natural-language output. Used for titles, descriptions, social copy."""
    host = _host_backend()
    if host is not None:
        return host._llm_prose(prompt, client or openrouter_client(),
                               max_tokens=max_tokens)
    return _chain(prompt, client, max_tokens)


def classify(prompt: str, client=None, max_tokens: int = 4000) -> str:
    """JSON output. The caller parses; this only guarantees fences are stripped."""
    host = _host_backend()
    if host is not None:
        return host._llm_classify(prompt, client or openrouter_client())
    prompt += "\n\nReturn only valid JSON, no markdown fences."
    return _chain(prompt, client, max_tokens)


def classify_json(prompt: str, client=None) -> dict:
    """classify(), plus fence-stripping and a parse. Raises on unparseable output."""
    raw = classify(prompt, client)
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        # Models sometimes wrap valid JSON in a sentence. Salvage the outer object.
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            return json.loads(match.group(0))
        raise LLMError(f"model did not return JSON: {exc}\n{raw[:400]}") from exc
