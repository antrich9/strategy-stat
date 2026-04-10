#!/usr/bin/env python3
import pandas as pd
import numpy as np

IN_FILE  = '/home/guiuser/.openclaw/workspace/output/final_research_catalog_with_backtests.csv'
OUT_FILE = '/home/guiuser/.openclaw/workspace/output/ranked_strategies_by_pair_timeframe.csv'

ALL_SYMBOLS = ['FX:GER40', 'FX:US30', 'FX:GBPUSD', 'FX:XAUUSD', 'FX:GER30', 'BINANCE:BTCUSDT']
MIN_TRADES = {'15': 300, '60': 250, '240': 150, 'D': 60}
MIN_PF     = 2.0

df = pd.read_csv(IN_FILE, low_memory=False)

# ── is_candidate ─────────────────────────────────────────────────────────────
df['is_candidate'] = (
    df['symbol'].isin(ALL_SYMBOLS) &
    (df['profit_factor'].astype(float).fillna(0) >= MIN_PF) &
    (df['trades'].astype(float).fillna(0) >= df['timeframe'].map(MIN_TRADES).astype(float))
)

cand = df[df['is_candidate']].copy()

# ── Normalise ────────────────────────────────────────────────────────────────
def normalize_g(g):
    mn, mx = g.min(), g.max()
    return (g - mn) / (mx - mn) if mx != mn else pd.Series(1.0, index=g.index)

cand['PF_norm']     = cand.groupby(['symbol','timeframe'])['profit_factor'].transform(normalize_g)
cand['DD_norm']    = cand.groupby(['symbol','timeframe'])['max_drawdown'].transform(
    lambda g: normalize_g(g.max() - g))
cand['Trades_norm']= cand.groupby(['symbol','timeframe'])['trades'].transform(normalize_g)

cand['StrategyScore'] = 0.45*cand['PF_norm'] + 0.35*cand['DD_norm'] + 0.20*cand['Trades_norm']

# ── top-pct helper ──────────────────────────────────────────────────────────
def top_pct(g, pct=0.10):
    t = g.quantile(1 - pct) if len(g) >= 10 else g.max()
    return g >= t

# 3 scopes
cand['is_top_10pct_pair_tf'] = cand.groupby(['symbol','timeframe'])['StrategyScore'].transform(
    lambda g: top_pct(g, 0.10))
cand['is_top_10pct_pair']    = cand.groupby('symbol')['StrategyScore'].transform(
    lambda g: top_pct(g, 0.10))
global_t = cand['StrategyScore'].quantile(0.90) if len(cand) >= 10 else cand['StrategyScore'].max()
cand['is_top_10pct_global']   = cand['StrategyScore'] >= global_t

# ── Write results back by index ────────────────────────────────────────────
df.loc[cand.index, 'PF_norm']           = cand['PF_norm']
df.loc[cand.index, 'DD_norm']           = cand['DD_norm']
df.loc[cand.index, 'Trades_norm']       = cand['Trades_norm']
df.loc[cand.index, 'StrategyScore']     = cand['StrategyScore']
df.loc[cand.index, 'is_top_10pct_pair_tf'] = cand['is_top_10pct_pair_tf']
df.loc[cand.index, 'is_top_10pct_pair']    = cand['is_top_10pct_pair']
df.loc[cand.index, 'is_top_10pct_global']  = cand['is_top_10pct_global']

df.to_csv(OUT_FILE, index=False)

print(f"Written: {OUT_FILE}")
print(f"Total rows: {len(df)}")
print(f"Candidates: {df['is_candidate'].sum()}")
print(f"Top 10% (pair+TF): {df['is_top_10pct_pair_tf'].sum()}")
print(f"Top 10% (pair):    {df['is_top_10pct_pair'].sum()}")
print(f"Top 10% (global):  {df['is_top_10pct_global'].sum()}")
print()
print("is_top_10pct_pair_tf by symbol+TF:")
print(df[df['is_top_10pct_pair_tf']==True].groupby(['symbol','timeframe']).size().to_string())
print()
print("is_top_10pct_pair by symbol:")
print(df[df['is_top_10pct_pair']==True].groupby('symbol').size().to_string())
print()
print("Top global strategies:")
cols = ['symbol','timeframe','csv_name','win_rate','profit_factor','max_drawdown','trades','StrategyScore']
cols = [c for c in cols if c in df.columns]
print(df[df['is_top_10pct_global']==True].sort_values('StrategyScore', ascending=False)[cols].head(15).to_string(index=False))