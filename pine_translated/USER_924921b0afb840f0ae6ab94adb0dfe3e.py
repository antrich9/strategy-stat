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
    # Wilder ATR implementation
    def wilder_atr(high, low, close, length=200):
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        atr = np.zeros_like(tr)
        atr[length - 1] = np.mean(tr[:length])
        for i in range(length, len(tr)):
            atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length
        return atr

    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    atr = wilder_atr(high, low, close, 200)

    # Filter width (default 0.0 from input)
    filterWidth = 0.0

    # Bullish FVG: low[3] > high[1] and close[2] < low[3] and close > low[3]
    # Need at least 4 bars to compute all offsets
    n = len(df)
    bull_signal = pd.Series(False, index=df.index)
    bear_signal = pd.Series(False, index=df.index)

    for i in range(3, n):
        if np.isnan(atr[i]):
            continue
        if high[i - 1] < low[i - 3] and close[i - 2] < low[i - 3] and close[i] > low[i - 3]:
            gap = low[i - 3] - high[i - 1]
            if gap - atr[i] * filterWidth > 0:
                bull_signal.iloc[i] = True

    for i in range(3, n):
        if np.isnan(atr[i]):
            continue
        if low[i - 1] > high[i - 3] and close[i - 2] > high[i - 3] and close[i] < high[i - 3]:
            gap = low[i - 1] - high[i - 3]
            if gap - atr[i] * filterWidth > 0:
                bear_signal.iloc[i] = True

    entries = []
    trade_num = 1

    for i in df.index:
        if bull_signal.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        if bear_signal.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries