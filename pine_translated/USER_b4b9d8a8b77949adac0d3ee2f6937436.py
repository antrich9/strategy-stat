import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Shifted series needed for Fair Value Gap detection
    high_2 = df['high'].shift(2)
    low_2 = df['low'].shift(2)
    close_1 = df['close'].shift(1)
    open_2 = df['open'].shift(2)
    open_1 = df['open'].shift(1)
    close_2 = df['close'].shift(2)

    # Bullish Fair Value Gap condition
    bull_fvg = (
        (df['low'] > high_2) &
        (close_1 > high_2) &
        (open_2 < close_2) &
        (open_1 < close_1) &
        (df['open'] < df['close'])
    ).fillna(False)

    # Bearish Fair Value Gap condition
    bear_fvg = (
        (df['high'] < low_2) &
        (close_1 < low_2) &
        (open_2 > close_2) &
        (open_1 > close_1) &
        (df['open'] > df['close'])
    ).fillna(False)

    # Extract hour for time‑of‑day filter (UTC)
    ts_index = pd.to_datetime(df['time'], unit='s', utc=True)
    hour = ts_index.hour
    valid_time = ((hour >= 2) & (hour < 5)) | ((hour >= 10) & (hour < 12))

    # Combine valid time with FVG signals
    long_cond = valid_time & bull_fvg
    short_cond = valid_time & bear_fvg

    entries = []
    trade_num = 1

    for i in df.index:
        if long_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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
        elif short_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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

    return entries