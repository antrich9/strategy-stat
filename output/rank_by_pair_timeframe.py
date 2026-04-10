#!/usr/bin/env python3
import pandas as pd
import numpy as np

IN_FILE  = '/home/guiuser/.openclaw/workspace/output/final_research_catalog_with_backtests.csv'
OUT_FILE = '/home/guiuser/.openclaw/workspace/output/ranked_strategies_by_pair_timeframe.csv'

ALLOWED_SYMBOLS = ['FX:GER40', 'FX:US30', 'FX:GBPUSD', 'FX:XAUUSD', 'FX:GER30']
EXCLUDED_SYMBOL = 'BINANCE:BTCUSDT'

MIN_TRADES = {'15': 300, '60': 250, '240': 150, 'D': 60}
MIN_PF     = 2.0

df = pd.read_csv(IN_FILE, low_memory=False)

df['is_candidate'] = (
    df['symbol'].isin(ALLOWED_SYMBOLS) &
    (df['profit_factor'].astype(float).fillna(0) >= MIN_PF) &
    (df['trades'].astype(float).fillna(0) >= df['timeframe'].map(MIN_TRADES).astype(float))
)

candidates = df[df['is_candidate']].copy()

def normalize_col(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series.map(lambda x: 1.0)
    return (series - mn) / (mx - mn)

def group_normalize(group_df, col, higher_is_better=True):
    if higher_is_better:
        normed = normalize_col(group_df[col])
    else:
        normed = normalize_col(group_df[col].max() - group_df[col])
    return normed

candidates['PF_norm']     = candidates.groupby(['symbol','timeframe'])['profit_factor'].transform(
    lambda g: normalize_col(g)
)
candidates['DD_norm']      = candidates.groupby(['symbol','timeframe'])['max_drawdown'].transform(
    lambda g: normalize_col(g.max() - g)
)
candidates['Trades_norm'] = candidates.groupby(['symbol','timeframe'])['trades'].transform(
    lambda g: normalize_col(g)
)

candidates['StrategyScore'] = (
    0.45 * candidates['PF_norm'] +
    0.35 * candidates['DD_norm'] +
    0.20 * candidates['Trades_norm']
)

def top10pct_flag(scores):
    if len(scores) < 10:
        threshold = scores.max()
    else:
        threshold = scores.quantile(0.90)
    return scores >= threshold

candidates['is_top_10pct'] = candidates.groupby(['symbol','timeframe'])['StrategyScore'].transform(
    top10pct_flag
)

df = df.merge(
    candidates[['key_norm','symbol','timeframe',
                'is_candidate','PF_norm','DD_norm','Trades_norm',
                'StrategyScore','is_top_10pct']],
    on=['key_norm','symbol','timeframe'],
    how='left'
)

for col in ['is_candidate','PF_norm','DD_norm','Trades_norm','StrategyScore','is_top_10pct']:
    if col not in df.columns:
        df[col] = np.nan

df.to_csv(OUT_FILE, index=False)
print(f"Written: {OUT_FILE}")
print(f"Total rows: {len(df)}")
print(f"Candidates: {df['is_candidate'].sum()}")
print(f"Top 10%: {df['is_top_10pct'].sum()}")