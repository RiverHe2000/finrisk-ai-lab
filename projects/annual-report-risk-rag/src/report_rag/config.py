"""Runtime settings, loaded from environment variables / ``.env``."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    """Configuration for retrieval and LLM calls.

    Every field can be overridden with a ``REPORT_RAG_`` prefixed environment
    variable, e.g. ``REPORT_RAG_MODEL=claude-sonnet-5``.
    """

    model_config = SettingsConfigDict(env_prefix="REPORT_RAG_", env_file=".env", extra="ignore")

    model: str = Field(default="claude-opus-5", description="Claude model id for extraction.")
    max_tokens: int = Field(default=4096, ge=256, description="Output cap per extraction call.")
    effort: str = Field(default="medium", description="Claude effort level for extraction.")
    chunk_size: int = Field(default=220, ge=50, description="Approx. tokens (words) per chunk.")
    chunk_overlap: int = Field(default=40, ge=0, description="Overlap in words between chunks.")
    top_k: int = Field(default=5, ge=1, description="Chunks passed to the extractor per metric.")
    bm25_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="Hybrid fusion weight.")
    data_dir: Path = Field(default=DEFAULT_DATA_DIR, description="Reports + gold labels root.")

    @property
    def reports_dir(self) -> Path:
        """Directory containing source annual reports."""
        return self.data_dir / "reports"

    @property
    def gold_dir(self) -> Path:
        """Directory containing gold-label JSON files."""
        return self.data_dir / "gold"
