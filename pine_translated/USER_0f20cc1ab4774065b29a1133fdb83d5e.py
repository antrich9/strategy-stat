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
    # Helper functions to identify OB and FVG conditions
    def is_up(close_arr, open_arr, idx):
        return close_arr[idx] > open_arr[idx]
    
    def is_down(close_arr, open_arr, idx):
        return close_arr[idx] < open_arr[idx]
    
    def is_ob_up(close_arr, open_arr, high_arr, low_arr, idx):
        return is_down(close_arr, open_arr, idx + 1) and is_up(close_arr, open_arr, idx) and close_arr[idx] > high_arr[idx + 1]
    
    def is_ob_down(close_arr, open_arr, high_arr, low_arr, idx):
        return is_up(close_arr, open_arr, idx + 1) and is_down(close_arr, open_arr, idx) and close_arr[idx] < low_arr[idx + 1]
    
    def is_fvg_up(low_arr, high_arr, idx):
        return low_arr[idx] > high_arr[idx + 2]
    
    def is_fvg_down(high_arr, low_arr, idx):
        return high_arr[idx] < low_arr[idx + 2]
    
    close = df['close'].values
    open_arr = df['open'].values
    high = df['high'].values
    low = df['low'].values
    time = df['time'].values
    volume = df['volume'].values
    
    n = len(df)
    
    # Initialize OB and FVG arrays
    ob_up = np.zeros(n, dtype=bool)
    ob_down = np.zeros(n, dtype=bool)
    fvg_up = np.zeros(n, dtype=bool)
    fvg_down = np.zeros(n, dtype=bool)
    
    # Calculate OB and FVG conditions (offset by 1 for obUp/obDown, 0 for fvgUp/fvgDown)
    for i in range(2, n - 2):
        ob_up[i] = is_ob_up(close, open_arr, high, low, i)
        ob_down[i] = is_ob_down(close, open_arr, high, low, i)
        fvg_up[i] = is_fvg_up(low, high, i)
        fvg_down[i] = is_fvg_down(high, low, i)
    
    # Wilder RSI implementation
    def wilder_rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(len(close))
        avg_loss = np.zeros(len(close))
        
        avg_gain[length] = np.mean(gain[1:length+1])
        avg_loss[length] = np.mean(loss[1:length+1])
        
        for i in range(length + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length - 1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length - 1) + loss[i]) / length
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Wilder ATR implementation
    def wilder_atr(high, low, close, length=14):
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        
        atr = np.zeros(len(high))
        atr[length] = np.mean(tr[1:length+1])
        
        for i in range(length + 1, len(high)):
            atr[i] = (atr[i-1] * (length - 1) + tr[i]) / length
        
        return atr
    
    # Calculate indicators
    sma_volume = pd.Series(volume).ewm(span=9, adjust=False).mean().values
    
    volfilt = np.ones(n, dtype=bool)  # Volume filter disabled
    atr = wilder_atr(high, low, close, 20) / 1.5
    atrfilt = np.ones(n, dtype=bool)  # ATR filter disabled
    
    loc = pd.Series(close).ewm(span=54, adjust=False).mean().values
    loc2 = loc > np.roll(loc, 1)
    loc2[0] = False
    locfiltb = loc2.copy()
    locfilts = ~loc2.copy()
    
    # Calculate bull/bear FVG signals
    bfvg = np.zeros(n, dtype=bool)
    sfvg = np.zeros(n, dtype=bool)
    
    for i in range(2, n - 2):
        bfvg[i] = low[i] > high[i + 2] and volfilt[i] and atrfilt[i] and locfiltb[i]
        sfvg[i] = high[i] < low[i + 2] and volfilt[i] and atrfilt[i] and locfilts[i]
    
    # Stacked OB+FVG conditions (entries)
    long_entry = ob_up & fvg_up
    short_entry = ob_down & fvg_down
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(n):
        if np.isnan(close[i]):
            continue
        
        direction = None
        if long_entry[i]:
            direction = 'long'
        elif short_entry[i]:
            direction = 'short'
        
        if direction:
            entry_ts = int(time[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price_guess = float(close[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1
    
    return entries