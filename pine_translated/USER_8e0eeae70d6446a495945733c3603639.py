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
    # Compute ATR (200) using Wilder's method (not directly used but kept for completeness)
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=200, adjust=False).mean()

    # --- Bullish VI ---
    bull_gap_top = np.minimum(close, df['open'])
    bull_gap_btm = np.maximum(close.shift(1), df['open'].shift(1))

    bull_vi_cond = (
        (df['open'] > close.shift(1)) &
        (high.shift(1) > low) &
        (close > close.shift(1)) &
        (df['open'] > df['open'].shift(1)) &
        (high.shift(1) < bull_gap_top)
    )

    # --- Bearish VI ---
    bear_gap_top = np.minimum(close.shift(1), df['open'].shift(1))
    bear_gap_btm = np.maximum(close, df['open'])

    bear_vi_cond = (
        (df['open'] < close.shift(1)) &
        (low.shift(1) < high) &
        (close < close.shift(1)) &
        (df['open'] < df['open'].shift(1)) &
        (low.shift(1) > bear_gap_btm)
    )

    # --- Bullish OG ---
    bull_og_cond = low > high.shift(1)

    # --- Bearish OG ---
    bear_og_cond = high < low.shift(1)

    # --- Bullish FVG ---
    bull_fvg_cond = (
        (low > high.shift(2)) &
        (close.shift(1) > high.shift(2)) &
        (df['open'].shift(2) < close.shift(2)) &
        (df['open'].shift(1) < close.shift(1)) &
        (df['open'] < close)
    )

    # --- Bearish FVG ---
    bear_fvg_cond = (
        (high < low.shift(2)) &
        (close.shift(1) < low.shift(2)) &
        (df['open'].shift(2) > close.shift(2)) &
        (df['open'].shift(1) > close.shift(1)) &
        (df['open'] > close)
    )

    # Fill NaNs with False for safe boolean indexing
    bull_vi = bull_vi_cond.fillna(False)
    bear_vi = bear_vi_cond.fillna(False)
    bull_og = bull_og_cond.fillna(False)
    bear_og = bear_og_cond.fillna(False)
    bull_fvg = bull_fvg_cond.fillna(False)
    bear_fvg = bear_fvg_cond.fillna(False)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        # Long entries
        if bull_fvg.iloc[i] or bull_og.iloc[i] or bull_vi.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = df['time'].iloc[i]
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

        # Short entries
        if bear_fvg.iloc[i] or bear_og.iloc[i] or bear_vi.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = df['time'].iloc[i]
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

    return entries