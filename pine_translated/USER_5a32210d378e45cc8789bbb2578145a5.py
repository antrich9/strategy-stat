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
    trade_num = 1
    
    # Extract data
    times = df['time'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    n = len(df)
    if n < 5:
        return results
    
    # Detect new days (start of each day at midnight UTC)
    dates = np.array([datetime.fromtimestamp(t, tz=timezone.utc).date() for t in times])
    is_new_day = np.zeros(n, dtype=bool)
    is_new_day[1:] = dates[1:] != dates[:-1]
    
    # Previous day high/low (shifted by 1 to get previous day values)
    prev_day_high = pd.Series(highs).shift(1).fillna(method='bfill').values
    prev_day_low = pd.Series(lows).shift(1).fillna(method='bfill').values
    
    # Asia session detection (2300-0700 London time)
    hours = np.array([datetime.fromtimestamp(t, tz=timezone.utc).hour for t in times])
    in_asia_session = ((hours >= 23) | (hours < 7))
    
    # Track Asia session high/low
    asia_high = np.full(n, np.nan)
    asia_low = np.full(n, np.nan)
    
    current_asia_high = np.nan
    current_asia_low = np.nan
    in_session_prev = False
    
    for i in range(n):
        is_new_sess = in_asia_session[i] and not in_session_prev
        
        if is_new_sess:
            current_asia_high = highs[i]
            current_asia_low = lows[i]
        elif in_asia_session[i]:
            current_asia_high = max(current_asia_high, highs[i]) if not np.isnan(current_asia_high) else highs[i]
            current_asia_low = min(current_asia_low, lows[i]) if not np.isnan(current_asia_low) else lows[i]
        
        if in_asia_session[i]:
            asia_high[i] = current_asia_high
            asia_low[i] = current_asia_low
        
        in_session_prev = in_asia_session[i]
    
    # Asia session high/low at session end
    asia_high_plot = pd.Series(asia_high).fillna(method='ffill').values
    asia_low_plot = pd.Series(asia_low).fillna(method='ffill').values
    
    # Detect sweeps
    asia_high_swept = highs > asia_high_plot
    asia_low_swept = lows < asia_low_plot
    
    # PDH/PDL sweep flags
    flag_pdh = np.zeros(n, dtype=bool)
    flag_pdl = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if is_new_day[i]:
            flag_pdh[i] = False
            flag_pdl[i] = False
        else:
            flag_pdh[i] = flag_pdh[i-1]
            flag_pdl[i] = flag_pdl[i-1]
        
        if closes[i] > prev_day_high[i]:
            flag_pdh[i] = True
        if closes[i] < prev_day_low[i]:
            flag_pdl[i] = True
    
    # OB/FVG detection functions
    def is_up(idx):
        if idx < 0 or idx >= n:
            return False
        return closes[idx] > opens[idx]
    
    def is_down(idx):
        if idx < 0 or idx >= n:
            return False
        return closes[idx] < opens[idx]
    
    def is_ob_up(idx):
        if idx - 1 < 0 or idx >= n:
            return False
        return is_down(idx - 1) and is_up(idx) and closes[idx] > highs[idx - 1]
    
    def is_ob_down(idx):
        if idx - 1 < 0 or idx >= n:
            return False
        return is_up(idx - 1) and is_down(idx) and closes[idx] < lows[idx - 1]
    
    def is_fvg_up(idx):
        if idx - 2 < 0 or idx >= n:
            return False
        return lows[idx] > highs[idx - 2]
    
    def is_fvg_down(idx):
        if idx - 2 < 0 or idx >= n:
            return False
        return highs[idx] < lows[idx - 2]
    
    # Detect stacked OB+FVG conditions
    ob_up = np.zeros(n, dtype=bool)
    ob_down = np.zeros(n, dtype=bool)
    fvg_up = np.zeros(n, dtype=bool)
    fvg_down = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        ob_up[i] = is_ob_up(i - 1)
        ob_down[i] = is_ob_down(i - 1)
        fvg_up[i] = is_fvg_up(i)
        fvg_down[i] = is_fvg_down(i)
    
    # Time filter checks (0700-0959 and 1200-1459)
    def check_time_filter(hour):
        in_first_window = 7 <= hour <= 9
        in_second_window = 12 <= hour <= 14
        return in_first_window or in_second_window
    
    # Entry conditions: PDH/PDL sweep + stacked OB/FVG
    long_condition = flag_pdh & ob_up & fvg_up
    short_condition = flag_pdl & ob_down & fvg_down
    
    # Generate entries
    for i in range(n):
        if np.isnan(closes[i]):
            continue
        
        direction = None
        if long_condition[i] and check_time_filter(hours[i]):
            direction = 'long'
        elif short_condition[i] and check_time_filter(hours[i]):
            direction = 'short'
        
        if direction:
            entry_ts = int(times[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(closes[i])
            
            results.append({
                'trade_num': trade_num,
                'direction': direction,
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
    
    return results