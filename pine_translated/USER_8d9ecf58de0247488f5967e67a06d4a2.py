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
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']

    fastLen = 8
    medLen = 20
    slowLen = 50

    ema8 = close.ewm(span=fastLen, adjust=False).mean()
    ema20 = close.ewm(span=medLen, adjust=False).mean()
    ema50 = close.ewm(span=slowLen, adjust=False).mean()

    dojiPerc = 0.30

    body = (close - open_).abs()
    rng = high - low
    isDoji = (rng > 0) & (body / rng <= dojiPerc)

    emaBull = (ema8 > ema20) & (ema20 > ema50)
    emaBear = (ema8 < ema20) & (ema20 < ema50)

    closeInUpper33 = (close - low) / rng >= 0.67
    closeInLower33 = (high - close) / rng >= 0.67

    bodyInUpper33 = (open_ >= low + rng * 0.67) & (close >= low + rng * 0.67)
    bodyInLower33 = (open_ <= low + rng * 0.33) & (close <= low + rng * 0.33)

    bullSignal = emaBull & isDoji & (low <= ema8) & closeInUpper33 & bodyInUpper33
    bearSignal = emaBear & isDoji & (high >= ema8) & closeInLower33 & bodyInLower33

    bullOnly = bullSignal & ~bearSignal
    bearOnly = bearSignal & ~bullSignal

    entries = []
    trade_num = 1
    position_open = False

    for i in range(1, len(df)):
        if position_open:
            continue

        if bullOnly.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])

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
            position_open = True
        elif bearOnly.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])

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
            position_open = True

    return entries