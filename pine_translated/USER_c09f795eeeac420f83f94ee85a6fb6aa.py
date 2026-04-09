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
    
    high = df['high']
    low = df['low']
    close = df['close']
    time = df['time']
    
    n = len(df)
    if n < 3:
        return []
    
    # Detect Fair Value Gaps (FVG)
    # Bullish FVG: low >= high[2] (current low is above the high from 2 bars ago)
    bullish_fvg = low >= high.shift(2)
    
    # Bearish FVG: high <= low[2] (current high is below the low from 2 bars ago)
    bearish_fvg = high <= low.shift(2)
    
    # London session time windows (UTC)
    # Morning: 08:00 - 09:55
    # Afternoon: 14:00 - 16:55
    london_morning_start_hour = 8
    london_morning_end_hour = 10
    london_afternoon_start_hour = 14
    london_afternoon_end_hour = 17
    
    def is_within_london_window(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        # Morning window: 08:00 - 09:55
        if hour == 8 or (hour == 9 and minute < 55):
            return True
        # Afternoon window: 14:00 - 16:55
        if (hour == 14) or (hour == 15) or (hour == 16 and minute < 55):
            return True
        return False
    
    # Track active FVG zones: list of dicts with 'top', 'bottom', 'detection_bar'
    bullish_fvg_zones = []
    bearish_fvg_zones = []
    
    entries = []
    trade_num = 1
    
    # Need at least 2 bars of history for FVG detection
    for i in range(2, n):
        current_time = time.iloc[i]
        
        # Check if within London session
        if not is_within_london_window(current_time):
            continue
        
        # Detect new FVGs on current bar
        # Bullish FVG: low >= high[2]
        if bullish_fvg.iloc[i]:
            zone_top = high.iloc[i-2]
            zone_bottom = low.iloc[i]
            if not np.isnan(zone_top) and not np.isnan(zone_bottom):
                bullish_fvg_zones.append({
                    'top': zone_top,
                    'bottom': zone_bottom,
                    'detection_bar': i
                })
        
        # Bearish FVG: high <= low[2]
        if bearish_fvg.iloc[i]:
            zone_top = high.iloc[i]
            zone_bottom = low.iloc[i-2]
            if not np.isnan(zone_top) and not np.isnan(zone_bottom):
                bearish_fvg_zones.append({
                    'top': zone_top,
                    'bottom': zone_bottom,
                    'detection_bar': i
                })
        
        # Check for long entries: price enters bullish FVG from above
        # Long entry when low touches/penetrates FVG top
        for zone in bullish_fvg_zones[:]:
            # Entry condition: low < zone_top (price enters FVG from above)
            # But we need to ensure we're not entering below the zone
            if low.iloc[i] < zone['top'] and high.iloc[i-1] >= zone['top']:
                entry_price = zone['top']
                entry_time_str = datetime.fromtimestamp(current_time, tz=timezone.utc).isoformat()
                
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(current_time),
                    'entry_time': entry_time_str,
                    'entry_price_guess': float(entry_price),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(entry_price),
                    'raw_price_b': float(entry_price)
                })
                trade_num += 1
                bullish_fvg_zones.remove(zone)
        
        # Check for short entries: price enters bearish FVG from below
        # Short entry when high touches/penetrates FVG bottom
        for zone in bearish_fvg_zones[:]:
            # Entry condition: high > zone_bottom (price enters FVG from below)
            if high.iloc[i] > zone['bottom'] and low.iloc[i-1] <= zone['bottom']:
                entry_price = zone['bottom']
                entry_time_str = datetime.fromtimestamp(current_time, tz=timezone.utc).isoformat()
                
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(current_time),
                    'entry_time': entry_time_str,
                    'entry_price_guess': float(entry_price),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(entry_price),
                    'raw_price_b': float(entry_price)
                })
                trade_num += 1
                bearish_fvg_zones.remove(zone)
    
    return entries