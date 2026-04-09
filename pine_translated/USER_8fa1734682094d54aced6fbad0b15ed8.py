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
    
    # Need at least 5 bars for swing detection (indices 0-4)
    if len(df) < 5:
        return []
    
    # Extract series
    high = df['high']
    low = df['low']
    close = df['close']
    time = df['time']
    
    # Swing Detection
    # Swing High: main_bar_high (high[2]) > high[1] and main_bar_high > high[3] and main_bar_high > high[4]
    is_swing_high = (high.shift(2) > high.shift(1)) & (high.shift(2) > high.shift(3)) & (high.shift(2) > high.shift(4))
    
    # Swing Low: main_bar_low (low[2]) < low[1] and main_bar_low < low[3] and main_bar_low < low[4]
    is_swing_low = (low.shift(2) < low.shift(1)) & (low.shift(2) < low.shift(3)) & (low.shift(2) < low.shift(4))
    
    # Store last swing high and low (forward filled)
    last_swing_high = is_swing_high * high.shift(2)
    last_swing_high = last_swing_high.replace(0, np.nan).ffill()
    
    last_swing_low = is_swing_low * low.shift(2)
    last_swing_low = last_swing_low.replace(0, np.nan).ffill()
    
    # FVG Detection
    # Bullish FVG: low[2] >= high (fair value gap to upside)
    bullish_fvg = low.shift(2) >= high
    
    # Bearish FVG: low >= high[2] (fair value gap to downside)
    bearish_fvg = low >= high.shift(2)
    
    # Entry conditions
    # Long: Bullish FVG exists at this bar
    long_condition = bullish_fvg
    
    # Short: Bearish FVG exists at this bar
    short_condition = bearish_fvg
    
    entries = []
    trade_num = 1
    
    for i in range(5, len(df)):
        # Long entry
        if long_condition.iloc[i]:
            timestamp = int(time.iloc[i])
            entry_price = float(close.iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': timestamp,
                'entry_time': datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        # Short entry
        if short_condition.iloc[i]:
            timestamp = int(time.iloc[i])
            entry_price = float(close.iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': timestamp,
                'entry_time': datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries