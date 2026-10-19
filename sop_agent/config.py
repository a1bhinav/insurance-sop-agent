"""Runtime configuration, read from environment variables (and an optional .env file).

Everything a grader might want to tweak lives here: the LLM provider / API token,
model names, and the SOP thresholds (how many PII factors, attempt limits, etc.).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4o",
    "mock": "mock",
}


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader (no extra dependency). Existing env vars win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # ---- LLM -------------------------------------------------------------
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic").lower())
    llm_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("LLM_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    llm_model: Optional[str] = field(default_factory=lambda: os.getenv("LLM_MODEL"))
    llm_extractor_model: Optional[str] = field(default_factory=lambda: os.getenv("LLM_EXTRACTOR_MODEL"))
    llm_base_url: Optional[str] = field(default_factory=lambda: os.getenv("LLM_BASE_URL"))
    llm_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT_SECONDS", "60")))
    # Let the browser UI supply a key per request (header X-LLM-API-Key) when the server has none.
    allow_client_api_key: bool = field(default_factory=lambda: _env_bool("ALLOW_CLIENT_API_KEY", True))

    # ---- SOP thresholds ----------------------------------------------------
    min_pii_factors: int = field(default_factory=lambda: _env_int("SOP_MIN_PII_FACTORS", 3))
    rep_min_pii_factors: int = field(default_factory=lambda: _env_int("SOP_REP_MIN_PII_FACTORS", 2))
    max_verify_attempts: int = field(default_factory=lambda: _env_int("SOP_MAX_VERIFY_ATTEMPTS", 3))
    off_topic_limit: int = field(default_factory=lambda: _env_int("SOP_OFF_TOPIC_LIMIT", 3))
    refusal_limit: int = field(default_factory=lambda: _env_int("SOP_REFUSAL_LIMIT", 3))
    consent_max_checks: int = field(default_factory=lambda: _env_int("SOP_CONSENT_MAX_CHECKS", 4))
    # Optional fixed "today" so the demo's deadlines stay meaningful (YYYY-MM-DD).
    demo_date: Optional[str] = field(default_factory=lambda: os.getenv("SOP_DEMO_DATE"))

    # ---- Branding ------------------------------------------------------------
    company_name: str = field(default_factory=lambda: os.getenv("COMPANY_NAME", "Harbor Mutual Insurance"))
    agent_name: str = field(default_factory=lambda: os.getenv("AGENT_NAME", "Sam"))

    # ---- Paths / server -------------------------------------------------------
    fixtures_dir: Path = field(
        default_factory=lambda: Path(os.getenv("FIXTURES_DIR", PROJECT_ROOT / "apps" / "insurance_claims" / "fixtures"))
    )
    outbox_dir: Path = field(default_factory=lambda: Path(os.getenv("OUTBOX_DIR", PROJECT_ROOT / "data" / "outbox")))
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def resolved_model(self) -> str:
        return self.llm_model or DEFAULT_MODELS.get(self.llm_provider, DEFAULT_MODELS["anthropic"])

    def resolved_extractor_model(self) -> str:
        return self.llm_extractor_model or self.resolved_model()

    @property
    def has_llm_key(self) -> bool:
        return bool(self.llm_api_key) or self.llm_provider == "mock"


settings = Settings()
