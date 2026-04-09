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
    # Shifted series for FVG detection
    low_2 = df['low'].shift(2)
    open_1 = df['open'].shift(1)
    high_0 = df['high']
    close_1 = df['close'].shift(1)
    close_0 = df['close']
    low_1 = df['low'].shift(1)

    # Bearish FVG (Top Imbalance) detection
    top_imbalance_size = low_2 - high_0
    top_imb_xbway = (low_2 <= open_1) & (high_0 >= close_1) & (close_0 > low_1)

    # Bullish FVG (Bottom Imbalance) detection
    high_2 = df['high'].shift(2)
    low_0 = df['low']
    high_1 = df['high'].shift(1)

    bottom_imbalance_size = low_0 - high_2
    bottom_imb_xbway = (high_2 >= open_1) & (low_0 <= close_1) & (close_0 < high_1)

    # Signal series (assuming FVGBKWY_on = true, FVGnew = true, enter_trades = true)
    short_signal = (top_imb_xbway & (top_imbalance_size > 0)).fillna(False)
    long_signal = (bottom_imb_xbway & (bottom_imbalance_size > 0)).fillna(False)

    # Trend detection using EMA200
    ema200 = df['close'].ewm(span=200, adjust=False).mean()
    is_uptrend = (df['close'] > ema200).fillna(False)
    is_downtrend = (df['close'] < ema200).fillna(False)

    # Entry signals combining FVG and trend
    long_entry = long_signal & is_uptrend
    short_entry = short_signal & is_downtrend

    # Build list of entries
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if long_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = close_0.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = close_0.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
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