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
    # Extract price series
    open_px = df['open']
    high_px = df['high']
    low_px = df['low']
    close_px = df['close']

    # Identify up/down candles
    is_up = close_px > open_px
    is_down = close_px < open_px

    # Detect Fair Value Gaps (FVG)
    bullish_fvg = low_px > high_px.shift(1)   # bullish gap: low > prior high
    bearish_fvg = high_px < low_px.shift(1)   # bearish gap: high < prior low

    # Detect Order Blocks (OB)
    # Bullish OB: a down candle immediately followed by an up candle
    bullish_ob = is_down.shift(1) & is_up
    # Bearish OB: an up candle immediately followed by a down candle
    bearish_ob = is_up.shift(1) & is_down

    # Stacked OB + FVG signals
    bull_stack = bullish_fvg & bullish_ob
    bear_stack = bearish_fvg & bearish_ob

    # Build entry list
    entries = []
    trade_num = 1

    for i in range(len(df)):
        # Skip bars where both stacked conditions are NaN (no signal)
        if pd.isna(bull_stack.iloc[i]) and pd.isna(bear_stack.iloc[i]):
            continue

        if bull_stack.iloc[i]:
            entry_price = close_px.iloc[i]
            entry_ts = int(df['time'].iloc[i])
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
        elif bear_stack.iloc[i]:
            entry_price = close_px.iloc[i]
            entry_ts = int(df['time'].iloc[i])
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