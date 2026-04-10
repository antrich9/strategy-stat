#!/usr/bin/env python3
"""
BTCUSDT 4H Shortlist Builder — modular version
Run: .venv/bin/python merge_and_shortlist.py
"""
import os, re, pandas as pd

WORKSPACE = '/home/guiuser/.openclaw/workspace'
CATALOG_PATH = '/home/guiuser/.openclaw/workspace-secondary/final_research_catalog.csv'
STATS_PATH   = f'{WORKSPACE}/strategy_stats.csv'
OUT_DIR      = f'{WORKSPACE}/output'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
SYMBOL     = "BINANCE:BTCUSDT"
TIMEFRAME  = "240"
MIN_TRADES = 30
MIN_WR     = 35
MIN_PF     = 1.0
MAX_DD     = 50
SCORE_COLS = ['csv_name','score','win_rate','profit_factor','max_drawdown','trades',
              'reach_1r_count','reach_2r_count','reach_3r_count',
              'core_type','family','logic_family','semantic_archetype_normalized',
              'stop_basis','trailing_basis','take_profit_basis',
              'semantic_bias_logic','semantic_entry_trigger','semantic_stop_loss_logic',
              'semantic_trailing_logic','semantic_suitable_markets','semantic_weaknesses',
              'semantic_risk_model','symbol','timeframe']

# ── Load data ────────────────────────────────────────────────────────────────
catalog = pd.read_csv(CATALOG_PATH)
stats   = pd.read_csv(STATS_PATH)

# ── Normalize merge key ────────────────────────────────────────────────────
def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r'[^a-z0-9]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

stats['key_norm']   = stats['strategy_name'].apply(norm)
catalog['key_norm'] = catalog['csv_name'].apply(norm)

# ── Merge: left join stats onto catalog ────────────────────────────────────
perf_cols = ['win_rate','profit_factor','max_drawdown','trades',
             'reach_1r_count','reach_2r_count','reach_3r_count']
sem_cols  = [c for c in catalog.columns
             if c not in ('script_id','file_name','csv_name','csv_version','key_norm')]

merged = catalog.merge(
    stats[['key_norm','symbol','timeframe'] + perf_cols],
    on='key_norm', how='left'
)

print(f"Merged: {len(merged)} rows | "
      f"{merged['win_rate'].notna().sum()} have backtest data | "
      f"{merged['win_rate'].isna().sum()} no backtest match")

merged.to_csv(f'{OUT_DIR}/final_research_catalog_with_backtests.csv', index=False)
print(f"Saved: {OUT_DIR}/final_research_catalog_with_backtests.csv")

# ── BTCUSDT 4H Shortlist ────────────────────────────────────────────────────
btc = merged[
    (merged['symbol'] == SYMBOL) &
    (merged['timeframe'] == TIMEFRAME) &
    (merged['trades'].notna()) &
    (merged['trades'] > 0)
].copy()
print(f"\n{SYMBOL} {TIMEFRAME}: {len(btc)} rows with trades>0")

# Filters
f1 = btc['trades'] >= MIN_TRADES
f2 = btc['win_rate'] >= MIN_WR
f3 = btc['profit_factor'] >= MIN_PF
f4 = btc['max_drawdown'] <= MAX_DD

# Trend/breakout archetypes
trend_kw = ['breakout','trend','fvg','momentum','continuation','donchian','structure']
atr_kw   = ['atr']

trend_mask = (
    btc['family'].str.lower().isin(trend_kw) |
    btc['logic_family'].str.lower().str.contains('|'.join(trend_kw), na=False) |
    btc['semantic_archetype_normalized'].str.lower().str.contains('|'.join(trend_kw), na=False)
)
atr_mask = (
    btc['stop_basis'].str.contains('atr', case=False, na=False) |
    btc['semantic_stop_loss_logic'].str.contains('atr', case=False, na=False)
)

btc_f = btc[f1 & f2 & f3 & f4 & trend_mask & atr_mask].copy()
print(f"After filters (trades>={MIN_TRADES}, WR>={MIN_WR}%, PF>={MIN_PF}, DD<={MAX_DD}%, trend/breakout, ATR stop): {len(btc_f)} rows")

# Score
btc_f['score'] = (
    btc_f['profit_factor'].astype(float) *
    (btc_f['win_rate'].astype(float) / 100) /
    (1 + btc_f['max_drawdown'].astype(float) / 100)
)
btc_f['score_robust'] = (
    btc_f['profit_factor'].astype(float) *
    (btc_f['win_rate'].astype(float) / 100) *
    (btc_f['trades'].astype(float).clip(upper=500) / 500) /
    (1 + btc_f['max_drawdown'].astype(float) / 100)
)

btc_ranked = btc_f.sort_values('score', ascending=False)

# Save
out_cols = [c for c in SCORE_COLS if c in btc_ranked.columns]
btc_ranked[out_cols].to_csv(f'{OUT_DIR}/btcusdt_4h_trend_shortlist.csv', index=False)
print(f"Saved: {OUT_DIR}/btcusdt_4h_trend_shortlist.csv")

# ── Print Top 20 ───────────────────────────────────────────────────────────
pd.set_option('display.max_colwidth', 40)
pd.set_option('display.float_format', '{:.3f}'.format)
show = ['csv_name','score','win_rate','profit_factor','max_drawdown','trades',
        'family','logic_family','semantic_archetype_normalized','stop_basis','trailing_basis']
show = [c for c in show if c in btc_ranked.columns]
print(f"\n{'='*90}")
print(f"TOP {min(20, len(btc_ranked))} — {SYMBOL} {TIMEFRAME}m | filters: trades>={MIN_TRADES}, WR>={MIN_WR}%, PF>={MIN_PF}, DD<={MAX_DD}%")
print(f"{'='*90}")
print(btc_ranked[show].head(20).to_string(index=False))

# ── Summary stats ───────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("SUMMARY")
print(f"{'='*90}")
print(f"Survivors: {len(btc_ranked)} rows | {btc_ranked['script_id'].nunique()} distinct strategies")
print(f"\nFamily distribution:\n{btc_ranked['family'].value_counts()}")
print(f"\nLogic family distribution:\n{btc_ranked['logic_family'].value_counts()}")
print(f"\nArchetype distribution:\n{btc_ranked['semantic_archetype_normalized'].value_counts()}")
print(f"\nStop basis:\n{btc_ranked['stop_basis'].value_counts()}")
print(f"\nTrailing basis:\n{btc_ranked['trailing_basis'].value_counts()}")
print(f"\nTake profit basis:\n{btc_ranked['take_profit_basis'].value_counts()}")

# Score distribution of survivors
print(f"\nScore stats: mean={btc_ranked['score'].mean():.3f}, max={btc_ranked['score'].max():.3f}")
print(f"Win rate stats: mean={btc_ranked['win_rate'].mean():.1f}%, range={btc_ranked['win_rate'].min():.1f}–{btc_ranked['win_rate'].max():.1f}%")
print(f"PF stats: mean={btc_ranked['profit_factor'].mean():.2f}, range={btc_ranked['profit_factor'].min():.2f}–{btc_ranked['profit_factor'].max():.2f}")
print(f"Trades: mean={btc_ranked['trades'].mean():.0f}, range={btc_ranked['trades'].min():.0f}–{btc_ranked['trades'].max():.0f}")
print(f"Max DD: mean={btc_ranked['max_drawdown'].mean():.1f}%, range={btc_ranked['max_drawdown'].min():.1f}–{btc_ranked['max_drawdown'].max():.1f}%")