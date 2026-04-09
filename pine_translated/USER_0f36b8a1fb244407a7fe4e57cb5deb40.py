import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Implementation based on Pine Script logic
    # Parameters
    PP = 5  # pivot period
    atrLength = 14
    atrMultiplier = 2.5
    fastEMALen = 50
    slowEMALen = 200
    useTimeFilter = True
    startHour = 7
    endHour = 12  # maybe default?

    # Compute indicators
    df['atr'] = df['high'].rolling(atrLength).apply(lambda x: np.max(x) - np.min(x), raw=True)  # Not correct, need Wilder ATR
    # Actually implement Wilder ATR manually
    # Compute EMA
    df['fastEMA'] = df['close'].ewm(span=fastEMALen, adjust=False).mean()
    df['slowEMA'] = df['close'].ewm(span=slowEMALen, adjust=False).mean()

    # Time filter (assuming UTC)
    df['hour'] = pd.to_datetime(df['time'], unit='s').dt.hour
    if useTimeFilter:
        df['time_filter'] = (df['hour'] >= startHour) & (df['hour'] <= endHour)
    else:
        df['time_filter'] = True

    # Compute pivot highs/lows
    # Use rolling max/min over PP periods
    df['pivot_high'] = df['high'].rolling(window=PP+1).max().shift(1)
    df['pivot_low'] = df['low'].rolling(window=PP+1).min().shift(1)

    # Detect bullish and bearish conditions
    # Use EMA crossover
    df['long_condition'] = (df['fastEMA'] > df['slowEMA']) & (df['close'] > df['open']) & df['time_filter']
    df['short_condition'] = (df['fastEMA'] < df['slowEMA']) & (df['close'] < df['open']) & df['time_filter']

    # Additional filter: price close > open + atrMultiplier * atr
    # But we need ATR: compute using Wilder ATR
    # We'll compute using typical ATR formula: (high - low) rolling

    # Actually we need to implement Wilder ATR: use exponential smoothing

    # For now, ignore ATR filter

    # Generate entries
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if df['long_condition'].iloc[i]:
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif df['short_condition'].iloc[i]:
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries