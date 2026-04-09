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
    # Precompute shifted series for higher‑timeframe values
    high = df['high']
    low = df['low']
    close = df['close']
    open_ = df['open']

    high2 = high.shift(2)
    low2 = low.shift(2)
    close1 = close.shift(1)
    open1 = open_.shift(1)

    entries = []
    fvg_top = None
    fvg_bottom = None
    fvg_created = False

    for i in range(len(df)):
        ts = int(df['time'].iloc[i])
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)

        # Session filter: 15:00‑15:59 GMT daily (all days)
        in_session = dt.hour == 15 and 0 <= dt.minute <= 59

        # bull_fvg detection
        if i >= 2:
            high2_i = high2.iloc[i]
            low_i = low.iloc[i]
            low2_i = low2.iloc[i]
            high_i = high.iloc[i]
            if pd.notna(high2_i) and pd.notna(low_i) and pd.notna(low2_i) and pd.notna(high_i):
                bull_fvg = (high2_i < low_i) or (low2_i > high_i)
            else:
                bull_fvg = False
        else:
            bull_fvg = False

        # Bullish candle on previous bar
        if i >= 1:
            close1_i = close1.iloc[i]
            open1_i = open1.iloc[i]
            if pd.notna(close1_i) and pd.notna(open1_i):
                is_bull = close1_i > open1_i
            else:
                is_bull = False
        else:
            is_bull = False

        # Detect the first FVG within the session
        if not fvg_created and bull_fvg and in_session:
            if is_bull:
                fvg_top = high2.iloc[i]    # _high2
                fvg_bottom = low.iloc[i]  # _low
            else:
                fvg_top = low2.iloc[i]    # _low2
                fvg_bottom = high.iloc[i] # _high
            fvg_created = True

        # Generate entries on retraces after FVG is established
        if fvg_created:
            close_i = close.iloc[i]
            if pd.isna(close_i):
                continue
            if is_bull and close_i <= fvg_bottom:
                trade_num = len(entries) + 1
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': dt.isoformat(),
                    'entry_price_guess': close_i,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close_i,
                    'raw_price_b': close_i
                })
            elif (not is_bull) and close_i >= fvg_top:
                trade_num = len(entries) + 1
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': dt.isoformat(),
                    'entry_price_guess': close_i,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close_i,
                    'raw_price_b': close_i
                })

    return entries