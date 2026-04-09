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
    
    # Parameters
    swingPeriod = 20
    atrLength = 14
    
    n = len(df)
    if n < 3:
        return []
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    timestamps = df['time'].values
    
    # Calculate ATR using Wilder's method
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr = np.zeros(n)
    atr[atrLength] = np.mean(tr[:atrLength])
    multiplier = 1.0 - 1.0 / atrLength
    for i in range(atrLength + 1, n):
        atr[i] = atr[i - 1] * multiplier + tr[i - 1]
    
    # Calculate swing high and low
    swingHigh = np.zeros(n)
    swingLow = np.zeros(n)
    for i in range(swingPeriod - 1, n):
        swingHigh[i] = np.max(high[i - swingPeriod + 1:i + 1])
        swingLow[i] = np.min(low[i - swingPeriod + 1:i + 1])
    
    # Detect FVGs
    # Bullish FVG (direction=1 in Pine): low[2] > high[0], returns [gap, low[2], high[0]]
    # bullFVG: gap detected, bullTop = high[0], bullBottom = low[2]
    bullFVG = np.zeros(n, dtype=bool)
    bullTop = np.zeros(n)
    bullBottom = np.zeros(n)
    
    # Bearish FVG (direction=-1 in Pine): high[2] < low[0], returns [gap, high[2], low[0]]
    # bearFVG: gap detected, bearTop = low[0], bearBottom = high[2]
    bearFVG = np.zeros(n, dtype=bool)
    bearTop = np.zeros(n)
    bearBottom = np.zeros(n)
    
    for i in range(2, n):
        # Bullish FVG: low from 2 bars ago > current high
        if low[i - 2] > high[i]:
            bullFVG[i] = True
            bullTop[i] = high[i]
            bullBottom[i] = low[i - 2]
        
        # Bearish FVG: high from 2 bars ago < current low
        if high[i - 2] < low[i]:
            bearFVG[i] = True
            bearTop[i] = low[i]
            bearBottom[i] = high[i - 2]
    
    # Check if FVG is in current range
    def in_current_range(idx):
        if idx < swingPeriod - 1:
            return False
        top = bullTop[idx] if bullFVG[idx] else bearTop[idx]
        bottom = bullBottom[idx] if bullFVG[idx] else bearBottom[idx]
        if np.isnan(top) or np.isnan(bottom):
            return False
        return (top <= swingHigh[idx] and bottom >= swingLow[idx]) or \
               (bottom >= swingLow[idx] and top <= swingHigh[idx]) or \
               (top >= swingHigh[idx] and bottom <= swingLow[idx])
    
    # Check if bar is within 3 PM to 4 PM London time
    in_trading_window = np.zeros(n, dtype=bool)
    for i in range(n):
        dt = datetime.fromtimestamp(timestamps[i], tz=timezone.utc)
        year = dt.year
        month = dt.month
        day = dt.day
        hour = dt.hour
        minute = dt.minute
        
        # Convert to London time (Europe/London)
        try:
            from datetime import timedelta
            london_tz_offset = 0 if (dt.month < 3 or dt.month > 10) or \
                                   (dt.month == 3 and dt.day - dt.weekday() <= 24 and (dt.weekday() > 0 or dt.day >= 25 - (7 - (dt.weekday() if dt.weekday() != 6 else 0)))) or \
                                   (dt.month == 10 and dt.day - dt.weekday() >= 24 and (dt.weekday() == 0 or dt.day < 25)) else 1
            if london_tz_offset == 0:
                london_dt = dt.replace(tzinfo=None) - dt.utcoffset()
            else:
                london_dt = dt.replace(tzinfo=None) - timedelta(hours=1)
        except:
            london_dt = dt
        
        # Check if within 15:00 to 16:00 London time
        if 15 <= london_dt.hour < 16:
            in_trading_window[i] = True
    
    # Reset at midnight (hour == 0 and minute == 0)
    is_midnight = np.zeros(n, dtype=bool)
    for i in range(n):
        dt = datetime.fromtimestamp(timestamps[i], tz=timezone.utc)
        if dt.hour == 0 and dt.minute == 0:
            is_midnight[i] = True
    
    # Track FVG state
    bullish_fvg_top = np.nan
    bullish_fvg_bottom = np.nan
    bearish_fvg_top = np.nan
    bearish_fvg_bottom = np.nan
    fvg_created = False
    
    entries = []
    trade_num = 1
    
    # Track if we have an open position (simulated for entry detection)
    in_position = False
    
    for i in range(3, n):
        # Reset FVG state at midnight
        if is_midnight[i]:
            bullish_fvg_top = np.nan
            bullish_fvg_bottom = np.nan
            bearish_fvg_top = np.nan
            bearish_fvg_bottom = np.nan
            fvg_created = False
            in_position = False
        
        # Create FVG boxes within trading window (3-4 PM)
        if in_trading_window[i] and not fvg_created:
            # Bullish FVG
            if bullFVG[i - 2] and in_current_range(i - 2) and np.isnan(bullish_fvg_top) and np.isnan(bullish_fvg_bottom):
                bullish_fvg_top = bullTop[i - 2]
                bullish_fvg_bottom = bullBottom[i - 2]
                fvg_created = True
            
            # Bearish FVG
            if bearFVG[i - 2] and in_current_range(i - 2) and np.isnan(bearish_fvg_top) and np.isnan(bearish_fvg_bottom):
                bearish_fvg_top = bearTop[i - 2]
                bearish_fvg_bottom = bearBottom[i - 2]
                fvg_created = True
        
        # Long entry: crossunder(low, bullishFvgTop)
        if in_trading_window[i] and not in_position and not np.isnan(bullish_fvg_bottom):
            if i > 0:
                prev_low = low[i - 1]
                curr_low = low[i]
                fvg_top = bullish_fvg_top
                if prev_low > fvg_top and curr_low <= fvg_top:
                    entry_time = datetime.fromtimestamp(timestamps[i], tz=timezone.utc).isoformat()
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': int(timestamps[i]),
                        'entry_time': entry_time,
                        'entry_price_guess': float(close[i]),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(close[i]),
                        'raw_price_b': float(close[i])
                    })
                    trade_num += 1
                    in_position = True
        
        # Short entry: crossover(high, bearishFvgBottom)
        if in_trading_window[i] and not in_position and not np.isnan(bearish_fvg_top):
            if i > 0:
                prev_high = high[i - 1]
                curr_high = high[i]
                fvg_bottom = bearish_fvg_bottom
                if prev_high < fvg_bottom and curr_high >= fvg_bottom:
                    entry_time = datetime.fromtimestamp(timestamps[i], tz=timezone.utc).isoformat()
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': int(timestamps[i]),
                        'entry_time': entry_time,
                        'entry_price_guess': float(close[i]),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(close[i]),
                        'raw_price_b': float(close[i])
                    })
                    trade_num += 1
                    in_position = True
        
        # Reset position flag when new day starts
        if is_midnight[i]:
            in_position = False
    
    return entries