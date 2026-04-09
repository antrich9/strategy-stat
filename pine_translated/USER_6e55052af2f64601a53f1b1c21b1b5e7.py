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
    # Price series
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']

    # Bullish / Bearish candles
    is_up = close > open_
    is_down = close < open_

    # Order block detection (isObUp/isObDown with index=1)
    is_ob_up = is_down.shift(2) & is_up.shift(1) & (close.shift(1) > high.shift(2))
    is_ob_down = is_up.shift(2) & is_down.shift(1) & (close.shift(1) < low.shift(2))

    # Fair value gap detection (isFvgUp/isFvgDown with index=0)
    is_fvg_up = low > high.shift(2)
    is_fvg_down = high < low.shift(2)

    # Stacked conditions: OB + FVG alignment
    bullish_stacked = is_ob_up & is_fvg_up
    bearish_stacked = is_ob_down & is_fvg_down

    # Fill NaNs with False for early bars lacking enough history
    bullish_stacked = bullish_stacked.fillna(False)
    bearish_stacked = bearish_stacked.fillna(False)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if bullish_stacked.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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
        elif bearish_stacked.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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