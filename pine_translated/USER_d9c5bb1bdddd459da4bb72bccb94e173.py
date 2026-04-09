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
    # Initialize variables
    open_col = df['open'].values
    high_col = df['high'].values
    low_col = df['low'].values
    close_col = df['close'].values
    time_col = df['time'].values
    
    n = len(df)
    entries = []
    trade_num = 1
    
    # Flags for PDH/PDL sweep detection
    flagpdl = False
    flagpdh = False
    
    # Store previous day high/low
    prevDayHigh = np.nan
    prevDayLow = np.nan
    
    # Detect new days and calculate previous day high/low
    # Convert time to datetime for day grouping
    times_dt = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in time_col]
    days = np.array([dt.date() for dt in times_dt])
    
    # Calculate previous day high/low for each bar
    pdh_values = np.full(n, np.nan)
    pdl_values = np.full(n, np.nan)
    
    unique_days = np.unique(days)
    for i, day in enumerate(unique_days[:-1]):
        next_day = unique_days[i + 1]
        day_mask = days == day
        if np.any(day_mask):
            prev_day_high = np.max(high_col[day_mask])
            prev_day_low = np.min(low_col[day_mask])
            # Apply to next day onwards until next unique day
            next_day_mask = days > day
            if np.any(next_day_mask):
                next_day_indices = np.where(next_day_mask)[0]
                pdh_values[next_day_indices] = prev_day_high
                pdl_values[next_day_indices] = prev_day_low
    
    # Calculate OB and FVG conditions
    # isUp: close > open (bullish candle)
    # isDown: close < open (bearish candle)
    # isObUp: isDown(i+1) and isUp(i) and close[i] > high[i+1]
    # isObDown: isUp(i+1) and isDown(i) and close[i] < low[i+1]
    # isFvgUp: low[i] > high[i+2]
    # isFvgDown: high[i] < low[i+2]
    
    obUp = np.full(n, False)
    obDown = np.full(n, False)
    fvgUp = np.full(n, False)
    fvgDown = np.full(n, False)
    
    for i in range(2, n):
        # OB conditions - need index i+1, so check bounds
        if i + 1 < n:
            is_current_up = close_col[i] > open_col[i]
            is_current_down = close_col[i] < open_col[i]
            is_prev_down = close_col[i + 1] < open_col[i + 1]
            is_prev_up = close_col[i + 1] > open_col[i + 1]
            
            # Bullish OB: previous candle is bearish, current is bullish, current close > previous high
            obUp[i] = is_prev_down and is_current_up and close_col[i] > high_col[i + 1]
            # Bearish OB: previous candle is bullish, current is bearish, current close < previous low
            obDown[i] = is_prev_up and is_current_down and close_col[i] < low_col[i + 1]
        
        # FVG conditions - need index i+2, so check bounds
        if i + 2 < n:
            # Bullish FVG: current low > high 2 bars ago
            fvgUp[i] = low_col[i] > high_col[i + 2]
            # Bearish FVG: current high < low 2 bars ago
            fvgDown[i] = high_col[i] < low_col[i + 2]
    
    # Detect sweeps and generate entries
    # Long entry: Bullish OB + FVG stacked AND previous day low was swept
    # Short entry: Bearish OB + FVG stacked AND previous day high was swept
    
    for i in range(3, n):
        current_pdh = pdh_values[i]
        current_pdl = pdl_values[i]
        
        # Check for sweeps
        if not np.isnan(current_pdh) and close_col[i] > current_pdh:
            flagpdh = True
        
        if not np.isnan(current_pdl) and close_col[i] < current_pdl:
            flagpdl = True
        
        # Generate long entry when flagpdl is true and bullish OB+FVG
        if flagpdl and obUp[i] and fvgUp[i]:
            entry_price = close_col[i]
            entry_ts = int(time_col[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            entries.append({
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
            trade_num += 1
            # Reset flag after entry
            flagpdl = False
        
        # Generate short entry when flagpdh is true and bearish OB+FVG
        if flagpdh and obDown[i] and fvgDown[i]:
            entry_price = close_col[i]
            entry_ts = int(time_col[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            entries.append({
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
            trade_num += 1
            # Reset flag after entry
            flagpdh = False
    
    return entries