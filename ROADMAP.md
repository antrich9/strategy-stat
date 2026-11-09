# Edge Ledger roadmap

Planned independent development: about 200 hours. Each milestone ships with tests and documentation.

## 1. Foundations (~20h)
- [ ] Make all paths configurable (no absolute machine paths); single config file
- [ ] Package layout (`edge_ledger/`), CLI entry point, `pyproject.toml`
- [ ] Test suite with pytest and sample public market data
- [ ] Continuous integration (lint + tests)

## 2. Experiment registry (~30h)
- [ ] Record every backtest run: strategy ID, parameters, symbol, timeframe, date range, cost assumptions, code version
- [ ] Store results (trades, equity curve, summary metrics) in SQLite
- [ ] Query and compare runs; never overwrite history
- [ ] Count trials per strategy family (needed for multiple-testing corrections)

## 3. Duplicate grouping (~25h)
- [ ] Detect near-identical strategies from their trade timing and return correlation
- [ ] Cluster duplicates so one idea tested many ways counts as one family
- [ ] Report the effective number of independent trials

## 4. Overfitting statistics (~40h)
- [ ] Deflated Sharpe Ratio (adjusts for number of trials, skew and kurtosis)
- [ ] Probability of Backtest Overfitting via combinatorially symmetric cross-validation
- [ ] Minimum track record length
- [ ] Multiple-testing corrections (Bonferroni, Holm, false discovery rate) on strategy p-values

## 5. Walk-forward validation (~35h)
- [ ] Rolling and anchored walk-forward splits
- [ ] Out-of-sample performance decay report (in-sample vs out-of-sample metrics)
- [ ] Regime splits (trending vs ranging, volatility buckets)
- [ ] Lookahead and repaint checks on translated strategies

## 6. Verdicts and reporting (~30h)
- [ ] One verdict per strategy family: robust / inconclusive / likely luck, with the reasons
- [ ] HTML report per run batch
- [ ] Track verdicts against later out-of-sample data to measure how well they hold up
- [ ] Replace `strategy_ranker.html` with a dashboard backed by the registry

## 7. Hardening (~20h)
- [ ] Performance: cached data loading, parallel runs
- [ ] Error handling and logging across the pipeline
- [ ] User guide and worked example notebook
