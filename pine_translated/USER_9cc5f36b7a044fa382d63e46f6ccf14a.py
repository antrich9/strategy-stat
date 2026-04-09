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
    
    results = []
    trade_num = 0
    
    n = len(df)
    if n < 6:
        return results
    
    # Extract series
    times = df['time'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    # Convert timestamps to datetime for time window filtering
    dts = [datetime.fromtimestamp(t, tz=timezone.utc) for t in times]
    hours = np.array([dt.hour for dt in dts])
    minutes = np.array([dt.minute for dt in dts])
    
    # London time window: 8:00-9:55 (morning) or 14:00-16:55 (afternoon)
    morning_window = ((hours == 8) | ((hours == 9) & (minutes < 55)))
    afternoon_window = ((hours == 14) | ((hours == 15) | ((hours == 16) & (minutes < 55))))
    is_within_time_window = morning_window | afternoon_window
    
    # FVG Detection on 15-minute timeframe
    # For simplicity, we simulate 15-min FVG using current TF bars
    # A bullish FVG occurs when low[2] >= high (gap up after drop)
    # A bearish FVG occurs when low >= high[2] (gap down after rise)
    
    bullish_fvg_top = np.full(n, np.nan)
    bullish_fvg_bottom = np.full(n, np.nan)
    bearish_fvg_top = np.full(n, np.nan)
    bearish_fvg_bottom = np.full(n, np.nan)
    
    # Detect FVGs
    for i in range(2, n):
        # Bullish FVG: low[2] >= high (previous candle high is within current low range)
        if lows[i-2] >= highs[i]:
            bullish_fvg_top[i] = lows[i-2]
            bullish_fvg_bottom[i] = highs[i]
        
        # Bearish FVG: low >= high[2] (previous candle low is within current high range)
        if lows[i] <= highs[i-2]:
            bearish_fvg_top[i] = highs[i-2]
            bearish_fvg_bottom[i] = lows[i]
    
    # Track active FVG boxes
    bull_boxes = []  # List of (top, bottom, created_idx)
    bear_boxes = []   # List of (top, bottom, created_idx)
    
    # Entry conditions
    for i in range(3, n):
        if not is_within_time_window[i]:
            continue
        
        # Clean up expired boxes (boxes older than 20 bars)
        bull_boxes = [(t, b, idx) for t, b, idx in bull_boxes if i - idx < 20]
        bear_boxes = [(t, b, idx) for t, b, idx in bear_boxes if i - idx < 20]
        
        # Add new FVG boxes
        if not np.isnan(bullish_fvg_top[i]):
            bull_boxes.append((bullish_fvg_top[i], bullish_fvg_bottom[i], i))
        
        if not np.isnan(bearish_fvg_top[i]):
            bear_boxes.append((bearish_fvg_top[i], bearish_fvg_bottom[i], i))
        
        current_low = lows[i]
        current_high = highs[i]
        
        # Check for bullish entries (price enters bullish FVG from above)
        for box_idx, (box_top, box_bottom, _) in enumerate(bull_boxes):
            if current_low < box_top:
                # Price entered bullish FVG
                trade_num += 1
                entry_price = closes[i]
                results.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(times[i]),
                    'entry_time': datetime.fromtimestamp(times[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                # Remove the box after entry
                bull_boxes[box_idx] = (np.nan, np.nan, -1)
                bull_boxes = [(t, b, idx) for t, b, idx in bull_boxes if not np.isnan(t)]
                break
        
        # Check for bearish entries (price enters bearish FVG from below)
        for box_idx, (box_top, box_bottom, _) in enumerate(bear_boxes):
            if current_high > box_bottom:
                # Price entered bearish FVG
                trade_num += 1
                entry_price = closes[i]
                results.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(times[i]),
                    'entry_time': datetime.fromtimestamp(times[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                # Remove the box after entry
                bear_boxes[box_idx] = (np.nan, np.nan, -1)
                bear_boxes = [(t, b, idx) for t, b, idx in bear_boxes if not np.isnan(t)]
                break
    
    return results