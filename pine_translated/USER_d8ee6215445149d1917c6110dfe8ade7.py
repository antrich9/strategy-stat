import pandas as pd
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

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
    
    close = df['close']
    high = df['high']
    low = df['low']
    open_arr = df['open']
    volume = df['volume']
    timestamps = df['time']
    
    n = len(df)
    
    # OB functions
    def is_ob_up(i):
        if i + 2 >= n:
            return False
        return (close.iloc[i+1] < open_arr.iloc[i+1] and 
                close.iloc[i] > open_arr.iloc[i] and 
                close.iloc[i] > high.iloc[i+1])
    
    def is_ob_down(i):
        if i + 2 >= n:
            return False
        return (close.iloc[i+1] > open_arr.iloc[i+1] and 
                close.iloc[i] < open_arr.iloc[i] and 
                close.iloc[i] < low.iloc[i+1])
    
    # FVG functions
    def is_fvg_up(i):
        if i + 2 >= n:
            return False
        return low.iloc[i] > high.iloc[i+2]
    
    def is_fvg_down(i):
        if i + 2 >= n:
            return False
        return high.iloc[i] < low.iloc[i+2]
    
    # Wilder RSI
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    # Wilder ATR
    def wilder_atr(high_arr, low_arr, close_arr, length):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - close_arr.shift(1))
        tr3 = np.abs(low_arr - close_arr.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr
    
    # RSI and ATR
    rsi = wilder_rsi(close, 14)
    atr = wilder_atr(high, low, close, 20)
    
    # Filters
    vol_ma = volume.rolling(9).mean()
    vol_filt = volume.shift(1) > vol_ma * 1.5
    
    atr_val = atr / 1.5
    low_minus_high2 = low - high.shift(2)
    high_minus_low2 = high - low.shift(2)
    atr_filt_long = low_minus_high2 > atr_val
    atr_filt_short = high_minus_low2 < -atr_val
    
    loc = close.ewm(span=54, adjust=False).mean()
    loc2 = loc > loc.shift(1)
    loc_filt_long = loc2
    loc_filt_short = ~loc2
    
    # Bullish/Bearish FVG
    bfvg = (low > high.shift(2)) & vol_filt & atr_filt_long & loc_filt_long
    sfvg = (high < low.shift(2)) & vol_filt & atr_filt_short & loc_filt_short
    
    # OB signals
    ob_up = pd.Series(False, index=df.index)
    ob_down = pd.Series(False, index=df.index)
    fvg_up = pd.Series(False, index=df.index)
    fvg_down = pd.Series(False, index=df.index)
    
    for i in range(2, n):
        ob_up.iloc[i] = is_ob_up(i)
        ob_down.iloc[i] = is_ob_down(i)
        fvg_up.iloc[i] = is_fvg_up(i)
        fvg_down.iloc[i] = is_fvg_down(i)
    
    # Stacked conditions
    long_cond = ob_up & fvg_up
    short_cond = ob_down & fvg_down
    
    # Time windows
    tz = ZoneInfo("Europe/Berlin")
    in_window = pd.Series(False, index=df.index)
    
    for i in range(n):
        try:
            dt_utc = datetime.fromtimestamp(timestamps.iloc[i] / 1000, tz=timezone.utc)
            dt_local = dt_utc.astimezone(tz)
            hour = dt_local.hour
            minute = dt_local.minute
            
            # First window: 07:00-10:59
            in_win1 = (7 <= hour <= 10) and not (hour == 10 and minute > 59)
            # Second window: 15:00-16:59
            in_win2 = (15 <= hour <= 16) and not (hour == 16 and minute > 59)
            
            in_window.iloc[i] = in_win1 or in_win2
        except:
            in_window.iloc[i] = False
    
    # Entry conditions
    long_entry = long_cond & in_window
    short_entry = short_cond & in_window
    
    # Generate entries
    for i in range(2, n):
        ts = int(timestamps.iloc[i])
        entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
        price = float(close.iloc[i])
        
        if long_entry.iloc[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
        
        if short_entry.iloc[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
    
    return results