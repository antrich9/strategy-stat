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
    # Ensure proper types
    df = df.copy()
    df['time'] = df['time'].astype(int)
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)

    # Determine trading session (London 03‑04 UTC, New York 10‑11 UTC)
    df['hour'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.hour
    is_trading_session = df['hour'].isin([3, 10])

    # Rolling highest high / lowest low (simulate 1h and 5m timeframe highest/lowest over 10 bars)
    high_rolling10 = df['high'].rolling(10).max()
    low_rolling10 = df['low'].rolling(10).min()

    # Market structure shift: crossover / crossunder with the rolling highest/lowest
    close = df['close']
    crossover_up = (close > high_rolling10) & (close.shift(1) <= high_rolling10.shift(1))
    crossunder_down = (close < low_rolling10) & (close.shift(1) >= low_rolling10.shift(1))

    # Fair Value Gap detection
    fvg_up = df['low'].shift(2) > df['high']
    fvg_down = df['high'].shift(2) < df['low']

    # Combined entry conditions
    long_cond = is_trading_session & crossover_up & fvg_up & (close > high_rolling10)
    short_cond = is_trading_session & crossunder_down & fvg_down & (close < low_rolling10)

    # Generate entry list
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries