# Edge Ledger (strategy-stat)

A robustness-testing toolkit for systematic trading research. It records every backtest run, groups duplicate strategies, and is being extended with walk-forward, Deflated Sharpe and overfitting tests, so that only strategies that are genuinely robust get traded, not ones that look good by luck across thousands of tests.

## Why it exists

I run a daily research pipeline that tests public, open-source TradingView strategies across several markets and timeframes. Out of thousands of tests, only around 7% of candidates pass full validation, and strategies that looked strong in backtests have disappointed live. The more strategies you test, the more "winners" appear by chance. Edge Ledger is the bookkeeping and statistics layer that corrects for that.

## Who it's for

Me, as a solo systematic trader. It should also be useful to anyone who batch-tests many strategies and needs to separate real edge from multiple-testing noise.

## What's in the repo today

| Path | What it does |
| --- | --- |
| `tv_pipeline.py` | Classifies Pine scripts into real strategies and runs standardised ATR / trailing-R backtests via a TradingView MCP connection |
| `batch_backtest.py`, `cached_backtest.py`, `simulate_standardized.py` | Batch and cached backtest runners across symbol/timeframe grids |
| `pine_translator.py` | Translates Pine Script entry logic into Python `generate_entries(df)` functions |
| `pine_translated/` | Python translations of open-source TradingView community strategies (see `NOTICE.md`) |
| `pine_templates/` | Strategy templates used for standardised testing |
| `output/` | Research catalogue, ranking scripts and shortlists from the first pipeline runs |
| `strategy_stats.csv`, `classified_strategies.json` | Results of the first backtest runs |
| `strategy_ranker.html` | A simple viewer for ranked results |

## Built with

Python 3 (pandas, NumPy), TradingView Desktop with an MCP bridge for chart backtests, and Pine Script. Developed with AI coding assistants.

## How to run (current state)

```bash
python3 -m venv .venv && .venv/bin/pip install pandas numpy
.venv/bin/python tv_pipeline.py classify    # find real strategies among the scripts
.venv/bin/python tv_pipeline.py backtest    # standardised backtests
.venv/bin/python batch_backtest.py          # parallel batch runs
```

Chart backtests need TradingView Desktop running with the MCP server. Some scripts still use absolute paths from my machine; making paths configurable is the first roadmap item.

## Roadmap

The Edge Ledger work (experiment registry, duplicate grouping, Deflated Sharpe, probability of backtest overfitting, walk-forward validation and verdict reports) is planned in [ROADMAP.md](ROADMAP.md).

## Licence

Mozilla Public License 2.0. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for third-party attribution.
