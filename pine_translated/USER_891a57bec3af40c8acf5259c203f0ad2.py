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
    n = len(df)
    entries = []
    trade_num = 1

    close = df['close']
    open_arr = df['open']
    high = df['high']
    low = df['low']

    # Bullish FVG: TopImbalance_Bway = low[2] <= open[1] and high[0] >= close[1] and close[0] < low[1]
    bull_fvg = (low.shift(2) <= open_arr.shift(1)) & (high >= close.shift(1)) & (close < low.shift(1))
    bull_fvg_size = low.shift(2) - high
    bull_fvg_valid = bull_fvg & (bull_fvg_size > 0)

    # Bearish FVG: BottomInbalance_Bway = high[2] >= open[1] and low[0] <= close[1] and close[0] > high[1]
    bear_fvg = (high.shift(2) >= open_arr.shift(1)) & (low <= close.shift(1)) & (close > high.shift(1))
    bear_fvg_size = low - high.shift(2)
    bear_fvg_valid = bear_fvg & (bear_fvg_size > 0)

    # Weak bullish FVG: Top_ImbXBway = low[2] <= open[1] and high[0] >= close[1] and close[0] > low[1]
    weak_bull_fvg = (low.shift(2) <= open_arr.shift(1)) & (high >= close.shift(1)) & (close > low.shift(1))
    weak_bull_fvg_size = low.shift(2) - high
    weak_bull_fvg_valid = weak_bull_fvg & (weak_bull_fvg_size > 0)

    # Weak bearish FVG: Bottom_ImbXBAway = high[2] >= open[1] and low[0] <= close[1] and close[0] < high[1]
    weak_bear_fvg = (high.shift(2) >= open_arr.shift(1)) & (low <= close.shift(1)) & (close < high.shift(1))
    weak_bear_fvg_size = low - high.shift(2)
    weak_bear_fvg_valid = weak_bear_fvg & (weak_bear_fvg_size > 0)

    # OB detection helpers
    isUp = close > open_arr
    isDown = close < open_arr

    # Bullish OB: isDown(index + 1) and isUp(index) - candle before is down, current is up
    bull_ob = isDown.shift(1) & isUp

    # Bearish OB: isUp(index + 1) and isDown(index) - candle before is up, current is down
    bear_ob = isUp.shift(1) & isDown

    # Combined signals - FVG with confirmation
    for i in range(3, n):
        # Long entry: bullish FVG detected at bar i-1 with confirmation at bar i
        if i >= 2 and (bull_fvg_valid.iloc[i-1] or weak_bull_fvg_valid.iloc[i-1]):
            if bull_ob.iloc[i]:
                ts = int(df['time'].iloc[i])
                entry_price = float(close.iloc[i])
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

        # Short entry: bearish FVG detected at bar i-1 with confirmation at bar i
        if i >= 2 and (bear_fvg_valid.iloc[i-1] or weak_bear_fvg_valid.iloc[i-1]):
            if bear_ob.iloc[i]:
                ts = int(df['time'].iloc[i])
                entry_price = float(close.iloc[i])
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