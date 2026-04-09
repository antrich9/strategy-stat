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
    # Core price series
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']

    # Shifted series for previous bars (Pine: dailyHigh[1], dailyHigh[2], etc.)
    high1 = high.shift(1)
    low1 = low.shift(1)
    high2 = high.shift(2)
    low2 = low.shift(2)
    high3 = high.shift(3)
    low3 = low.shift(3)
    high4 = high.shift(4)
    low4 = low.shift(4)

    # -------------------------------------------------
    # Filters (mirrors Pine inputs; defaults disabled)
    # Volume filter (inp1 = false → always true)
    vol_filt = pd.Series(True, index=df.index)

    # ATR filter (inp2 = false → always true) – Wilder ATR(20)
    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atr2 = atr / 1.5
    atrfilt = ((low - high2) > atr2) | ((low2 - high) > atr2)

    # Trend filter (inp3 = false → always true)
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    loc_filt_b = pd.Series(True, index=df.index)
    loc_filt_s = pd.Series(True, index=df.index)

    # -------------------------------------------------
    # Fair Value Gap detection
    bfvg = (low > high2) & vol_filt & atrfilt & loc_filt_b   # bullish FVG
    sfvg = (high < low2) & vol_filt & atrfilt & loc_filt_s   # bearish FVG

    # Swing detection (local high/low)
    is_swing_high = (high1 < high2) & (high3 < high2) & (high4 < high2)
    is_swing_low = (low1 > low2) & (low3 > low2) & (low4 > low2)

    # -------------------------------------------------
    # Iterate to generate entry signals
    entries = []
    trade_num = 1
    last_swing_type = None  # "dailyLow" or "dailyHigh"

    # Need at least 4 bars for the required shifts
    start_idx = 4
    for i in range(start_idx, len(df)):
        # Update most recent swing type
        if is_swing_high.iloc[i]:
            last_swing_type = "dailyHigh"
        if is_swing_low.iloc[i]:
            last_swing_type = "dailyLow"

        # Bullish entry
        if bfvg.iloc[i] and last_swing_type == "dailyLow":
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

        # Bearish entry
        elif sfvg.iloc[i] and last_swing_type == "dailyHigh":
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