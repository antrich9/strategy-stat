import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.

    Returns list of dicts:
    [{'trade_num': int, 'direction': 'long' or 'short',
      'entry_ts': int, 'entry_time': str,
      'entry_price_guess': float,
      'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
      'raw_price_a': float, 'raw_price_b': float}]
    """
    PP = 5
    atr_length = 55
    atr_length_sl = 14
    atr_multiplier = 1.5

    atr = df['high'].diff().abs().combine_first(df['low'].diff().abs()).combine_first((df['high'] - df['low']).abs())
    atr = atr.rolling(window=atr_length).mean()

    true_range = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    atr_sl = true_range.rolling(window=atr_length_sl).mean()

    pivot_high = pd.Series(np.nan, index=df.index)
    pivot_low = pd.Series(np.nan, index=df.index)

    for i in range(PP, len(df)):
        if df['high'].iloc[i] == df['high'].iloc[i-PP:i+1].max() and df['high'].iloc[i] > df['high'].iloc[i-PP:i].max():
            pivot_high.iloc[i] = df['high'].iloc[i]
        if df['low'].iloc[i] == df['low'].iloc[i-PP:i+1].min() and df['low'].iloc[i] < df['low'].iloc[i-PP:i].min():
            pivot_low.iloc[i] = df['low'].iloc[i]

    swing_highs = df.loc[pivot_high.notna(), 'high'].copy()
    swing_lows = df.loc[pivot_low.notna(), 'low'].copy()
    swing_high_idx = pivot_high[pivot_high.notna()].index
    swing_low_idx = pivot_low[pivot_low.notna()].index

    entries = []
    trade_num = 0

    major_boh_price = np.nan
    major_bol_price = np.nan
    major_boh_idx = -1
    major_bol_idx = -1
    minor_boh_price = np.nan
    minor_bol_price = np.nan
    minor_boh_idx = -1
    minor_bol_idx = -1
    last_high_idx = -1
    last_low_idx = -1
    last_high_price = np.nan
    last_low_price = np.nan

    for i in range(PP * 2 + 1, len(df)):
        if pivot_high.iloc[i] == pivot_high.iloc[i]:
            last_high_price = pivot_high.iloc[i]
            last_high_idx = i
        if pivot_low.iloc[i] == pivot_low.iloc[i]:
            last_low_price = pivot_low.iloc[i]
            last_low_idx = i

        if last_high_idx > last_low_idx and not np.isnan(major_bol_price):
            atr_val = atr.iloc[i] if atr.iloc[i] == atr.iloc[i] else 0
            if df['close'].iloc[i] < major_bol_price - atr_multiplier * atr_val:
                trade_num += 1
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(major_bol_price),
                    'raw_price_b': float(df['close'].iloc[i])
                })
                major_bol_price = np.nan

        if last_low_idx > last_high_idx and not np.isnan(major_boh_price):
            atr_val = atr.iloc[i] if atr.iloc[i] == atr.iloc[i] else 0
            if df['close'].iloc[i] > major_boh_price + atr_multiplier * atr_val:
                trade_num += 1
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(major_boh_price),
                    'raw_price_b': float(df['close'].iloc[i])
                })
                major_boh_price = np.nan

        if pivot_high.iloc[i] == pivot_high.iloc[i]:
            if len(swing_high_idx) >= 2:
                prev_high_idx = swing_high_idx[swing_high_idx.get_loc(i) - 1] if swing_high_idx.get_loc(i) > 0 else -1
                if prev_high_idx >= 0 and prev_high_idx > major_boh_idx:
                    major_boh_price = df['high'].iloc[prev_high_idx]
                    major_boh_idx = prev_high_idx
            if i - last_low_idx > PP * 2 and (major_bol_idx < 0 or last_low_idx > major_bol_idx):
                if not np.isnan(last_low_price):
                    major_bol_price = last_low_price
                    major_bol_idx = last_low_idx

        if pivot_low.iloc[i] == pivot_low.iloc[i]:
            if len(swing_low_idx) >= 2:
                prev_low_idx = swing_low_idx[swing_low_idx.get_loc(i) - 1] if swing_low_idx.get_loc(i) > 0 else -1
                if prev_low_idx >= 0 and prev_low_idx > major_bol_idx:
                    major_bol_price = df['low'].iloc[prev_low_idx]
                    major_bol_idx = prev_low_idx
            if i - last_high_idx > PP * 2 and (major_boh_idx < 0 or last_high_idx > major_boh_idx):
                if not np.isnan(last_high_price):
                    major_boh_price = last_high_price
                    major_boh_idx = last_high_idx

    return entries