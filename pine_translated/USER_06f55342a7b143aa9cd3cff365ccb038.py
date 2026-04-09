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
    if len(df) < 2:
        return []

    close = df['close']
    high = df['high']
    low = df['low']
    hl2 = (df['high'] + df['low']) / 2

    # SuperTrend Parameters
    Periods = 10
    Multiplier = 3.0

    # Calculate ATR (Wilder)
    tr = np.maximum(
        np.maximum(high - low, np.abs(high - close.shift(1))),
        np.abs(low - close.shift(1))
    )
    alpha = 1.0 / Periods
    atr = pd.Series(index=df.index, dtype=float)
    atr.iloc[0] = tr.iloc[0]
    for i in range(1, len(df)):
        atr.iloc[i] = (1 - alpha) * atr.iloc[i-1] + alpha * tr.iloc[i]

    # SuperTrend calculation
    up = hl2 - Multiplier * atr
    up1 = up.shift(1).fillna(up)
    up = np.where(close.shift(1) > up1, np.maximum(up, up1), up)

    dn = hl2 + Multiplier * atr
    dn1 = dn.shift(1).fillna(dn)
    dn = np.where(close.shift(1) < dn1, np.minimum(dn, dn1), dn)

    # Trend calculation
    trend = pd.Series(1.0, index=df.index)
    trend.iloc[0] = 1.0

    for i in range(1, len(df)):
        if trend.iloc[i-1] == -1 and close.iloc[i] > dn1.iloc[i]:
            trend.iloc[i] = 1.0
        elif trend.iloc[i-1] == 1 and close.iloc[i] < up1.iloc[i]:
            trend.iloc[i] = -1.0
        else:
            trend.iloc[i] = trend.iloc[i-1]

    # Buy/Sell signals
    trend_prev = trend.shift(1).fillna(1)
    buySignal = (trend == 1) & (trend_prev == -1)
    sellSignal = (trend == -1) & (trend_prev == 1)

    trade_num = 1
    entries = []

    for i in range(1, len(df)):
        entry_price = close.iloc[i]
        ts = df['time'].iloc[i]
        entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        if buySignal.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': entry_time_str,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

        if sellSignal.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
                'entry_time': entry_time_str,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

    return entries