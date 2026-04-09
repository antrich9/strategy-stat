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
    # Convert unix timestamps to datetime (UTC)
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.sort_values('datetime').reset_index(drop=True)

    # Resample to 4-hour bars
    df_4h = df.groupby(pd.Grouper(key='datetime', freq='4h', origin='start')).agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum')
    ).dropna().reset_index()

    # Lagged high/low (2 x 4‑hour periods ago)
    df_4h['high_lag2'] = df_4h['high'].shift(2)
    df_4h['low_lag2'] = df_4h['low'].shift(2)

    # Fair‑Value‑Gap flags on the 4‑hour timeframe
    df_4h['bullish_fvg'] = df_4h['low'] > df_4h['high_lag2']
    df_4h['bearish_fvg'] = df_4h['high'] < df_4h['low_lag2']

    last_fvg = 0  # 0 = none, 1 = bullish, -1 = bearish
    trade_num = 1
    entries = []

    # Iterate over 4‑hour bars (skip first two bars where lag is NaN)
    for i in range(2, len(df_4h)):
        row = df_4h.iloc[i]
        if pd.isna(row['high_lag2']) or pd.isna(row['low_lag2']):
            continue

        bull = bool(row['bullish_fvg'])
        bear = bool(row['bearish_fvg'])

        # Sharp‑turn long entry
        if bull and last_fvg == -1:
            entry_price = float(row['close'])
            entry_ts = int(row['datetime'].timestamp())
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        # Sharp‑turn short entry
        elif bear and last_fvg == 1:
            entry_price = float(row['close'])
            entry_ts = int(row['datetime'].timestamp())
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        # Update last FVG direction
        if bull:
            last_fvg = 1
        elif bear:
            last_fvg = -1

    return entries