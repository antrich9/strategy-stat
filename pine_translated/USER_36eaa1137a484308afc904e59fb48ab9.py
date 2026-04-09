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
    # Input parameters (fixed for this strategy)
    fastLength = 8
    mediumLength = 20
    slowLength = 50
    atrLength = 7

    # Compute EMAs
    close = df['close']
    fastEMA = close.ewm(span=fastLength, adjust=False).mean()
    mediumEMA = close.ewm(span=mediumLength, adjust=False).mean()
    slowEMA = close.ewm(span=slowLength, adjust=False).mean()

    # Compute True Range (Wilder's method)
    high = df['high']
    low = df['low']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = np.maximum(tr1, np.maximum(tr2, tr3)).fillna(tr1)

    # ATR (Wilder smoothed) – not needed for entries but computed for completeness
    atr = tr.ewm(alpha=1/atrLength, adjust=False).mean()

    # Crossover / Crossunder detection
    crossover = (close > fastEMA) & (close.shift(1) <= fastEMA.shift(1))
    crossunder = (close < fastEMA) & (close.shift(1) >= fastEMA.shift(1))

    # Entry conditions
    longCondition = crossover & (close > mediumEMA) & (close > slowEMA)
    shortCondition = crossunder & (close < mediumEMA) & (close < slowEMA)

    # Generate entry list
    entries = []
    trade_num = 1

    for i in range(len(df)):
        # Skip bars where any required indicator is NaN
        if (pd.isna(close.iloc[i]) or pd.isna(fastEMA.iloc[i]) or
            pd.isna(mediumEMA.iloc[i]) or pd.isna(slowEMA.iloc[i])):
            continue

        if longCondition.iloc[i]:
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
        elif shortCondition.iloc[i]:
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