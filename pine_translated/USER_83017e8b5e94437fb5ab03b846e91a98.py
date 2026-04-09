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
    
    # Extract columns
    times = df['time'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    opens = df['open'].values
    
    n = len(df)
    
    # Convert timestamps to datetime for time window filtering
    # London morning: 08:00-09:55 UTC
    # London afternoon: 14:00-16:55 UTC
    in_trading_window = np.zeros(n, dtype=bool)
    
    for i in range(n):
        dt = datetime.fromtimestamp(times[i], tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        total_minutes = hour * 60 + minute
        
        # Morning window: 480-595 (8:00-9:55)
        # Afternoon window: 840-1015 (14:00-16:55)
        in_morning = 480 <= total_minutes < 595
        in_afternoon = 840 <= total_minutes < 1015
        in_trading_window[i] = in_morning or in_afternoon
    
    # Detect FVGs (Fair Value Gaps)
    # Bullish FVG: low[2] >= high (gap up from 2 bars ago)
    # Bearish FVG: low >= high[2] (gap down)
    
    bullish_fvg_top = np.full(n, np.nan)
    bullish_fvg_bottom = np.full(n, np.nan)
    bullish_fvg_start_time = np.full(n, np.nan)
    
    bear_fvg_top = np.full(n, np.nan)
    bear_fvg_bottom = np.full(n, np.nan)
    bear_fvg_start_time = np.full(n, np.nan)
    
    # Track active FVG boxes
    # For each bar, detect if FVG formed 2 bars ago
    for i in range(4, n):
        # Bullish FVG: low[i-2] >= high[i] means gap up
        # The FVG zone is from low[i-2] to high[i]
        if lows[i-2] >= highs[i]:
            # Bullish FVG detected
            top = lows[i-2]
            bottom = highs[i]
            bullish_fvg_top[i] = top
            bullish_fvg_bottom[i] = bottom
            bullish_fvg_start_time[i] = times[i-2]
        
        # Bearish FVG: low[i] >= high[i-2] means gap down
        # The FVG zone is from low[i] to high[i-2]
        if lows[i] >= highs[i-2]:
            # Bearish FVG detected
            top = highs[i-2]
            bottom = lows[i]
            bear_fvg_top[i] = top
            bear_fvg_bottom[i] = bottom
            bear_fvg_start_time[i] = times[i-2]
    
    # Track active FVG boxes and generate entries
    active_bull_boxes = []  # list of dicts: {'top': float, 'bottom': float, 'start_time': int, 'entered': bool}
    active_bear_boxes = []  # list of dicts: {'top': float, 'bottom': float, 'start_time': int, 'entered': bool}
    
    for i in range(n):
        if not in_trading_window[i]:
            continue
        
        current_high = highs[i]
        current_low = lows[i]
        
        # Add newly detected FVGs as active boxes
        if not np.isnan(bullish_fvg_top[i]):
            active_bull_boxes.append({
                'top': bullish_fvg_top[i],
                'bottom': bullish_fvg_bottom[i],
                'start_time': int(bullish_fvg_start_time[i]),
                'entered': False
            })
        
        if not np.isnan(bear_fvg_top[i]):
            active_bear_boxes.append({
                'top': bear_fvg_top[i],
                'bottom': bear_fvg_bottom[i],
                'start_time': int(bear_fvg_start_time[i]),
                'entered': False
            })
        
        # Process bull boxes
        boxes_to_remove = []
        for idx, box in enumerate(active_bull_boxes):
            if box['entered']:
                continue
            
            box_bottom = box['bottom']
            box_top = box['top']
            
            # If price closes below box bottom, remove box
            if current_low < box_bottom:
                boxes_to_remove.append(idx)
            # If price enters the FVG zone (touches inside)
            elif current_low < box_top:
                # Long entry
                trade_num += 1
                entry_price = closes[i]
                entry_ts = int(times[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                
                results.append({
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
                box['entered'] = True
                boxes_to_remove.append(idx)
        
        # Remove processed boxes (reverse order to maintain indices)
        for idx in sorted(boxes_to_remove, reverse=True):
            if idx < len(active_bull_boxes):
                active_bull_boxes.pop(idx)
        
        # Process bear boxes
        boxes_to_remove = []
        for idx, box in enumerate(active_bear_boxes):
            if box['entered']:
                continue
            
            box_bottom = box['bottom']
            box_top = box['top']
            
            # If price closes above box top, remove box
            if current_high > box_top:
                boxes_to_remove.append(idx)
            # If price enters the FVG zone (touches inside)
            elif current_high > box_bottom:
                # Short entry
                trade_num += 1
                entry_price = closes[i]
                entry_ts = int(times[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                
                results.append({
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
                box['entered'] = True
                boxes_to_remove.append(idx)
        
        # Remove processed boxes (reverse order to maintain indices)
        for idx in sorted(boxes_to_remove, reverse=True):
            if idx < len(active_bear_boxes):
                active_bear_boxes.pop(idx)
    
    return results