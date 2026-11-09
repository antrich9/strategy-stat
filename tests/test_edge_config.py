import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ENV_VARS = (
    "EDGE_LEDGER_HOME",
    "EDGE_LEDGER_OUTPUT",
    "EDGE_LEDGER_CATALOG",
    "EDGE_LEDGER_CACHE",
    "EDGE_LEDGER_PYTHON",
)


def load_config(monkeypatch, **env):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    import edge_config
    return importlib.reload(edge_config)


def test_defaults_are_relative_to_repo(monkeypatch):
    cfg = load_config(monkeypatch)
    assert cfg.HOME == REPO_ROOT
    assert cfg.TRANSLATED_DIR == REPO_ROOT / "pine_translated"
    assert cfg.STATS_CSV == REPO_ROOT / "strategy_stats.csv"
    assert cfg.OUTPUT_DIR == REPO_ROOT / "output"
    assert cfg.CATALOG_CSV == REPO_ROOT / "output" / "final_research_catalog.csv"


def test_no_machine_specific_paths_in_defaults(monkeypatch):
    cfg = load_config(monkeypatch)
    for path in (cfg.HOME, cfg.TRANSLATED_DIR, cfg.STATS_CSV, cfg.OUTPUT_DIR, cfg.CATALOG_CSV):
        assert ".openclaw" not in str(path)


def test_home_override_moves_dependent_paths(monkeypatch, tmp_path):
    cfg = load_config(monkeypatch, EDGE_LEDGER_HOME=tmp_path)
    assert cfg.HOME == tmp_path.resolve()
    assert cfg.STATS_CSV == tmp_path.resolve() / "strategy_stats.csv"
    assert cfg.OUTPUT_DIR == tmp_path.resolve() / "output"


def test_output_and_catalog_overrides(monkeypatch, tmp_path):
    out = tmp_path / "results"
    catalog = tmp_path / "catalog.csv"
    cfg = load_config(monkeypatch, EDGE_LEDGER_OUTPUT=out, EDGE_LEDGER_CATALOG=catalog)
    assert cfg.OUTPUT_DIR == out.resolve()
    assert cfg.CATALOG_CSV == catalog.resolve()


def test_venv_python_falls_back_to_current_interpreter(monkeypatch, tmp_path):
    cfg = load_config(monkeypatch, EDGE_LEDGER_HOME=tmp_path)
    assert cfg.venv_python() == sys.executable


def test_venv_python_explicit_override(monkeypatch):
    cfg = load_config(monkeypatch, EDGE_LEDGER_PYTHON="/opt/py/bin/python3")
    assert cfg.venv_python() == "/opt/py/bin/python3"
