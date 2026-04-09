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
    # Keep series for quick access
    high = df['high']
    low = df['low']
    close = df['close']

    # ---------- Wilder ATR (period 144) ----------
    period = 144
    if len(df) < period:
        return []

    prev_close = close.shift(1)
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )

    atr = tr.rolling(period).mean()
    atr_vals = atr.values.copy()
    # Apply Wilder smoothing from bar 'period' onward
    for i in range(period, len(atr_vals)):
        if not np.isnan(atr_vals[i]):
            atr_vals[i] = (atr_vals[i - 1] * (period - 1) + tr.iloc[i]) / period
    atr = pd.Series(atr_vals, index=df.index)

    # Fair Value Gap width filter (default 0.5)
    fvg_th = 0.5
    atr_th = atr * fvg_th

    # ---------- Bull / Bear detection ----------
    bullG = low > high.shift(1)
    bearG = high < low.shift(1)

    bull = (
        (low - high.shift(2) > atr_th) &
        (low > high.shift(2)) &
        (close.shift(1) > high.shift(2)) &
        ~(bullG | bullG.shift(1))
    )

    bear = (
        (low.shift(2) - high > atr_th) &
        (high < low.shift(2)) &
        (close.shift(1) < low.shift(2)) &
        ~(bearG | bearG.shift(1))
    )

    bull = bull.fillna(False)
    bear = bear.fillna(False)

    # ---------- Build entry list ----------
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if bull.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif bear.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries