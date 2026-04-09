import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Ensure sorted by time
    df = df.sort_values('time').reset_index(drop=True)
    # Compute indicators
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']

    # Shifted series
    high_1 = high.shift(1)
    high_2 = high.shift(2)
    low_1 = low.shift(1)
    low_2 = low.shift(2)
    close_1 = close.shift(1)

    # Bullish FVG
    bull_fvg = (low > high_2) & (close_1 > high_2)
    # Bearish FVG
    bear_fvg = (high < low_2) & (close_1 < low_2)

    # Optionally compute filtered versions if needed (ignored for default)
    # volfilt, atrfilt, locfiltb, locfilts could be computed but omitted

    # Build list of entries
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if pd.isna(bull_fvg.iloc[i]) or pd.isna(bear_fvg.iloc[i]):
            continue
        if bull_fvg.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = close.iloc[i]
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
        elif bear_fvg.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = close.iloc[i]
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