"""
edge_config.py — central path configuration for the research pipeline.

Every script resolves its files through this module instead of hard-coded
machine paths, so the repo runs from any checkout location.

Defaults are relative to the repository root. Override with environment
variables when your data lives elsewhere:

    EDGE_LEDGER_HOME     base folder (default: repo root)
    EDGE_LEDGER_OUTPUT   output folder (default: <home>/output)
    EDGE_LEDGER_CATALOG  research catalogue CSV (default: <output>/final_research_catalog.csv)
    EDGE_LEDGER_CACHE    bar-data cache file (default: <system temp>/bars_cache.pkl)
    EDGE_LEDGER_PYTHON   interpreter for worker subprocesses (default: <home>/.venv python, else current)
"""
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


HOME = _env_path("EDGE_LEDGER_HOME", REPO_ROOT)
TRANSLATED_DIR = HOME / "pine_translated"
STATS_CSV = HOME / "strategy_stats.csv"
SYMBOLS_FILE = HOME / "symbols_timeframes.json"
OUTPUT_DIR = _env_path("EDGE_LEDGER_OUTPUT", HOME / "output")
CATALOG_CSV = _env_path("EDGE_LEDGER_CATALOG", OUTPUT_DIR / "final_research_catalog.csv")
CACHE_FILE = _env_path("EDGE_LEDGER_CACHE", Path(tempfile.gettempdir()) / "bars_cache.pkl")


def venv_python() -> str:
    """Interpreter used to launch worker subprocesses."""
    explicit = os.environ.get("EDGE_LEDGER_PYTHON")
    if explicit:
        return explicit
    candidate = HOME / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable
