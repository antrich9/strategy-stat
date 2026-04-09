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
    
    open_col = df['open'].values
    high_col = df['high'].values
    low_col = df['low'].values
    close_col = df['close'].values
    time_col = df['time'].values
    volume_col = df['volume'].values
    
    n = len(df)
    
    # Helper functions
    def is_up(idx):
        if idx < 0 or idx >= n:
            return False
        return close_col[idx] > open_col[idx]
    
    def is_down(idx):
        if idx < 0 or idx >= n:
            return False
        return close_col[idx] < open_col[idx]
    
    def is_ob_up(idx):
        # isDown(idx + 1) and isUp(idx) and close[idx] > high[idx + 1]
        if idx < 0 or idx + 1 >= n:
            return False
        return is_down(idx + 1) and is_up(idx) and close_col[idx] > high_col[idx + 1]
    
    def is_ob_down(idx):
        # isUp(idx + 1) and isDown(idx) and close[idx] < low[idx + 1]
        if idx < 0 or idx + 1 >= n:
            return False
        return is_up(idx + 1) and is_down(idx) and close_col[idx] < low_col[idx + 1]
    
    def is_fvg_up(idx):
        # low[idx] > high[idx + 2]
        if idx < 0 or idx + 2 >= n:
            return False
        return low_col[idx] > high_col[idx + 2]
    
    def is_fvg_down(idx):
        # high[idx] < low[idx + 2]
        if idx < 0 or idx + 2 >= n:
            return False
        return high_col[idx] < low_col[idx + 2]
    
    # Calculate OB and FVG conditions
    ob_up = np.zeros(n, dtype=bool)
    ob_down = np.zeros(n, dtype=bool)
    fvg_up = np.zeros(n, dtype=bool)
    fvg_down = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        ob_up[i] = is_ob_up(i - 1)  # obUp = isObUp(1)
        ob_down[i] = is_ob_down(i - 1)  # obDown = isObDown(1)
        fvg_up[i] = is_fvg_up(i - 0)  # fvgUp = isFvgUp(0)
        fvg_down[i] = is_fvg_down(i - 0)  # fvgDown = isFvgDown(0)
    
    # Calculate indicators for bfvg/sfvg conditions
    # volfilt = inp1 ? volume[1] > ta.sma(volume, 9)*1.5 : true
    # atr = ta.atr(20) / 1.5
    # loc = ta.sma(close, 54)
    # loc2 = loc > loc[1]
    
    # Volume filter
    vol_sma = pd.Series(volume_col).rolling(9).mean().values
    volfilt = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(vol_sma[i]):
            volfilt[i] = volume_col[i-1] > vol_sma[i] * 1.5
        else:
            volfilt[i] = True  # default true when inp1 is false
    
    # ATR calculation (Wilder)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high_col[i] - low_col[i], 
                    abs(high_col[i] - close_col[i-1]), 
                    abs(low_col[i] - close_col[i-1]))
    
    atr = np.zeros(n)
    atr[19] = tr[1:20].sum()  # First ATR is simple sum of first 19 TR values
    for i in range(20, n):
        atr[i] = (atr[i-1] * 19 + tr[i]) / 20
    
    atr_scaled = atr / 1.5
    
    # ATR filter: (low - high[2] > atr) or (low[2] - high > atr)
    atrfilt = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if not np.isnan(atr_scaled[i]):
            atrfilt[i] = (low_col[i] - high_col[i-2] > atr_scaled[i]) or (low_col[i-2] - high_col[i] > atr_scaled[i])
        else:
            atrfilt[i] = True  # default true when inp2 is false
    
    # Trend filter
    loc = pd.Series(close_col).rolling(54).mean().values
    loc2 = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(loc[i]) and not np.isnan(loc[i-1]):
            loc2[i] = loc[i] > loc[i-1]
        else:
            loc2[i] = False
    
    locfiltb = np.zeros(n, dtype=bool)
    locfilts = np.zeros(n, dtype=bool)
    for i in range(n):
        locfiltb[i] = True  # inp3 default false, so locfiltb always true
        locfilts[i] = True  # inp3 default false, so locfilts always true
    
    # bfvg and sfvg conditions
    bfvg = np.zeros(n, dtype=bool)
    sfvg = np.zeros(n, dtype=bool)
    for i in range(2, n):
        bfvg[i] = low_col[i] > high_col[i-2] and volfilt[i] and atrfilt[i] and locfiltb[i]
        sfvg[i] = high_col[i] < low_col[i-2] and volfilt[i] and atrfilt[i] and locfilts[i]
    
    # Trading time windows (London time)
    # Morning: 7:45 - 9:45, Afternoon: 14:45 - 16:45
    in_trading_window = np.zeros(n, dtype=bool)
    
    for i in range(n):
        ts = time_col[i]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        total_minutes = hour * 60 + minute
        
        # Morning window: 7:45 to 9:45 (465 to 585 minutes)
        morning_start = 7 * 60 + 45  # 465
        morning_end = 9 * 60 + 45    # 585
        
        # Afternoon window: 14:45 to 16:45 (885 to 1005 minutes)
        afternoon_start = 14 * 60 + 45  # 885
        afternoon_end = 16 * 60 + 45    # 1005
        
        in_morning = morning_start <= total_minutes < morning_end
        in_afternoon = afternoon_start <= total_minutes < afternoon_end
        in_trading_window[i] = in_morning or in_afternoon
    
    # Entry conditions: stacked OB + FVG within trading window
    # Long: ob_up and fvg_up
    # Short: ob_down and fvg_down
    long_condition = ob_up & fvg_up & in_trading_window
    short_condition = ob_down & fvg_down & in_trading_window
    
    entries = []
    trade_num = 1
    
    for i in range(n):
        if long_condition[i]:
            entry_ts = int(time_col[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close_col[i])
            
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
        
        if short_condition[i]:
            entry_ts = int(time_col[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close_col[i])
            
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
    
    return entries