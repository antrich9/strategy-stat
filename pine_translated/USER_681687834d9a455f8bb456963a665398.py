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
    if len(df) < 20:
        return []

    entries = []
    trade_num = 1

    pp = 5
    atr_length = 14

    high_pivot = df['high'].rolling(window=pp+1).max().shift(pp)
    low_pivot = df['low'].rolling(window=pp+1).min().shift(pp)

    swing_high = high_pivot.copy()
    swing_low = low_pivot.copy()

    for i in range(pp, len(df)):
        if pd.notna(high_pivot.iloc[i]):
            swing_high.iloc[i] = df['high'].iloc[i]
            for j in range(i-pp, i+1):
                if df['high'].iloc[j] > df['high'].iloc[i]:
                    swing_high.iloc[i] = df['high'].iloc[j]
                    break
        if pd.notna(low_pivot.iloc[i]):
            swing_low.iloc[i] = df['low'].iloc[i]
            for j in range(i-pp, i+1):
                if df['low'].iloc[j] < df['low'].iloc[i]:
                    swing_low.iloc[i] = df['low'].iloc[j]
                    break

    close_series = df['close']

    bull_mss = (close_series > close_series.shift(1)) & (close_series.shift(1) <= close_series.shift(2))
    bear_mss = (close_series < close_series.shift(1)) & (close_series.shift(1) >= close_series.shift(2))

    swing_high_shifted = swing_high.shift(1)
    swing_low_shifted = swing_low.shift(1)

    bull_choch = pd.Series(False, index=df.index)
    bear_choch = pd.Series(False, index=df.index)

    for i in range(pp + 5, len(df)):
        if pd.notna(swing_low.iloc[i]) and pd.notna(swing_low_shifted.iloc[i]):
            if df['close'].iloc[i] > swing_low_shifted.iloc[i] and df['close'].iloc[i-1] <= swing_low_shifted.iloc[i-1]:
                bull_choch.iloc[i] = True
        if pd.notna(swing_high.iloc[i]) and pd.notna(swing_high_shifted.iloc[i]):
            if df['close'].iloc[i] < swing_high_shifted.iloc[i] and df['close'].iloc[i-1] >= swing_high_shifted.iloc[i-1]:
                bear_choch.iloc[i] = True

    long_cond = bull_choch | bull_mss
    short_cond = bear_choch | bear_mss

    for i in range(pp + 5, len(df)):
        if pd.isna(swing_high.iloc[i]) or pd.isna(swing_low.iloc[i]):
            continue

        entry_price = df['close'].iloc[i]
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        if long_cond.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        if short_cond.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries