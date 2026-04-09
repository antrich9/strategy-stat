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
    
    # All filters are disabled by default in the original script
    # volfilt = volume[1] > ta.sma(volume, 9)*1.5 (disabled)
    # atrfilt = ((low - high[2] > atr) or (low[2] - high > atr)) (disabled)
    # locfiltb = loc > loc[1] where loc = ta.sma(close, 54) (disabled)
    # So all filters are True by default
    
    # Helper Series with proper shifts
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    volume_s = df['volume']
    
    # Open/Close comparisons for OB detection
    is_up = close > open_
    is_down = close < open_
    
    # Order Block conditions (using shift for previous bars)
    # isObUp(index): isDown(index + 1) and isUp(index) and close[index] > high[index + 1]
    # For current bar (index 0), we need isDown(1) and isUp(0) and close(0) > high(1)
    ob_up = is_down.shift(1) & is_up & (close > high.shift(1))
    
    # isObDown(index): isUp(index + 1) and isDown(index) and close[index] < low[index + 1]
    ob_down = is_up.shift(1) & is_down & (close < low.shift(1))
    
    # FVG conditions
    # isFvgUp(index): low[index] > high[index + 2]
    # For current bar: low[0] > high[2]
    fvg_up = low > high.shift(2)
    
    # isFvgDown(index): high[index] < low[index + 2]
    fvg_down = high < low.shift(2)
    
    # Stack conditions: obUp(1) means previous bar had OB pattern
    # fvgUp(0) means current bar has FVG
    # In Pine: obUp = isObUp(1), fvgUp = isFvgUp(0)
    stacked_bull = ob_up.shift(1) & fvg_up
    stacked_bear = ob_down.shift(1) & fvg_down
    
    # Also need isUp(0) for bullish (bullish candle for entry) and isDown(0) for bearish
    bull_entry_cond = stacked_bull & is_up
    bear_entry_cond = stacked_bear & is_down
    
    # Build entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if bull_entry_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif bear_entry_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries