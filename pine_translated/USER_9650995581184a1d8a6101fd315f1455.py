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
    if len(df) < 3:
        return []

    # Bull/Bear engulfing checks (bullG, bearG)
    bullG = df['close'] > df['open']
    bearG = df['high'] < df['low']

    # Bullish FVG: low > high[2] and not engulfing
    bull_fvg = (df['low'] > df['high'].shift(2)) & ~bullG & ~bullG.shift(1)

    # Bearish FVG: high < low[2] and not engulfing
    bear_fvg = (df['high'] < df['low'].shift(2)) & ~bearG & ~bearG.shift(1)

    # ATR threshold (Wilder method) - needed for FVG filtering
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atr_threshold = atr * 0.5

    # Full FVG conditions with ATR filtering
    bull_fvg_full = bull_fvg & ((df['low'] - df['high'].shift(2)) > atr_threshold)
    bear_fvg_full = bear_fvg & ((df['low'].shift(2) - df['high']) > atr_threshold)

    # Consecutive FVG counters
    consecutive_bull = np.zeros(len(df))
    consecutive_bear = np.zeros(len(df))
    last_fvg_bull = np.full(len(df), np.nan, dtype=object)

    for i in range(2, len(df)):
        if bull_fvg_full.iloc[i]:
            consecutive_bull[i] = consecutive_bull[i - 1] + 1
            consecutive_bear[i] = 0
            last_fvg_bull[i] = True
        elif bear_fvg_full.iloc[i]:
            consecutive_bull[i] = 0
            consecutive_bear[i] = consecutive_bear[i - 1] + 1
            last_fvg_bull[i] = False
        else:
            consecutive_bull[i] = 0
            consecutive_bear[i] = 0
            last_fvg_bull[i] = np.nan

    # FVG active check (at least one FVG exists in recent bars)
    fvg_active = (consecutive_bull > 0) | (consecutive_bear > 0)

    # Mitigation check: price has moved into the FVG zone
    # For bullish FVG (zone: low to high[2]), price should be at or below the top
    bull_mitigated = (df['low'] <= df['high'].shift(2)) & ((df['close'] <= df['high'].shift(2)) | (df['close'].shift(1) <= df['high'].shift(2)))
    # For bearish FVG (zone: low[2] to high), price should be at or above the bottom
    bear_mitigated = (df['high'] >= df['low'].shift(2)) & ((df['close'] >= df['low'].shift(2)) | (df['close'].shift(1) >= df['low'].shift(2)))

    # lastPct check: must be > 0.01 and <= 1 (using NaN-safe boolean)
    last_pct_valid = (~pd.isna(last_fvg_bull)) & (last_fvg_bull != np.nan)

    # Long entry conditions
    long_entry = fvg_active & last_pct_valid & (consecutive_bull == 2) & bull_mitigated

    # Short entry conditions
    short_entry = fvg_active & last_pct_valid & (consecutive_bear == 2) & bear_mitigated

    trades = []
    trade_num = 1

    for i in range(2, len(df)):
        if pd.isna(atr.iloc[i]) or pd.isna(last_fvg_bull.iloc[i]):
            continue

        if long_entry.iloc[i]:
            trades.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

        if short_entry.iloc[i]:
            trades.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return trades