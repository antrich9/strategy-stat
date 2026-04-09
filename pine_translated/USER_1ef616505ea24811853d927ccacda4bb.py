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
    
    open_vals = df['open']
    high_vals = df['high']
    low_vals = df['low']
    close_vals = df['close']
    volume_vals = df['volume']
    time_vals = df['time']
    
    # Helper functions for bar comparisons
    def is_up(idx):
        return close_vals.iloc[idx] > open_vals.iloc[idx]
    
    def is_down(idx):
        return close_vals.iloc[idx] < open_vals.iloc[idx]
    
    # FVG conditions
    fvg_up = low_vals > high_vals.shift(2)
    fvg_down = high_vals < low_vals.shift(2)
    
    # Volume filter: volume[1] > sma(volume, 9) * 1.5
    vol_sma9 = volume_vals.rolling(9).mean()
    vol_filt = volume_vals.shift(1) > vol_sma9 * 1.5
    
    # ATR filter: (low - high[2] > atr) or (low[2] - high > atr)
    high_low = high_vals - low_vals
    high_close_low = high_vals.shift(1) - low_vals
    close_low_high = close_vals.shift(1) - high_vals
    
    tr1 = high_low
    tr2 = high_close_low
    tr3 = close_low_high
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder ATR(20) using EMA with alpha = 1/20
    alpha = 1 / 20
    atr_wilder = true_range.ewm(alpha=alpha, adjust=False).mean()
    atr_threshold = atr_wilder / 1.5
    
    atrfilt_up = (low_vals - high_vals.shift(2) > atr_threshold) | (low_vals.shift(2) - high_vals > atr_threshold)
    atrfilt_down = (high_vals - low_vals.shift(2) > atr_threshold) | (high_vals.shift(2) - low_vals > atr_threshold)
    
    # Trend filter: sma(close, 54) > sma(close, 54)[1]
    loc_sma = close_vals.rolling(54).mean()
    loc_trend_up = loc_sma > loc_sma.shift(1)
    loc_trend_down = ~loc_trend_up
    
    # Bull/Bear FVG
    bfvg = fvg_up & vol_filt & atrfilt_up & loc_trend_up
    sfvg = fvg_down & vol_filt & atrfilt_down & loc_trend_down
    
    # London time windows (8:00-9:45 and 15:00-16:45)
    in_window = pd.Series([False] * len(df), index=df.index)
    
    for i in range(len(df)):
        ts = time_vals.iloc[i] / 1000.0  # Convert ms to seconds
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        
        morning_start = (hour == 8 and minute >= 0)
        morning_end = (hour == 9 and minute <= 45)
        morning = morning_start or (hour > 8 and hour < 9) or morning_end
        
        afternoon_start = (hour == 15 and minute >= 0)
        afternoon_end = (hour == 16 and minute <= 45)
        afternoon = afternoon_start or (hour > 15 and hour < 16) or afternoon_end
        
        in_window.iloc[i] = morning or afternoon
    
    # Stack OB + FVG conditions
    ob_up = pd.Series([False] * len(df), index=df.index)
    ob_down = pd.Series([False] * len(df), index=df.index)
    fvg_up_curr = pd.Series([False] * len(df), index=df.index)
    fvg_down_curr = pd.Series([False] * len(df), index=df.index)
    
    for i in range(2, len(df)):
        ob_up.iloc[i] = is_down(i - 1) and is_up(i) and close_vals.iloc[i] > high_vals.iloc[i - 1]
        ob_down.iloc[i] = is_up(i - 1) and is_down(i) and close_vals.iloc[i] < low_vals.iloc[i - 1]
        fvg_up_curr.iloc[i] = low_vals.iloc[i] > high_vals.iloc[i - 2]
        fvg_down_curr.iloc[i] = high_vals.iloc[i] < low_vals.iloc[i - 2]
    
    # Combined conditions
    long_cond = (bfvg & in_window) | (ob_up & fvg_up_curr & in_window)
    short_cond = (sfvg & in_window) | (ob_down & fvg_down_curr & in_window)
    
    trade_num = 1
    entries = []
    
    for i in range(len(df)):
        if pd.isna(close_vals.iloc[i]):
            continue
        
        entry_price = close_vals.iloc[i]
        entry_ts = int(time_vals.iloc[i])
        entry_time = datetime.fromtimestamp(entry_ts / 1000.0, tz=timezone.utc).isoformat()
        
        if long_cond.iloc[i]:
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
        
        if short_cond.iloc[i]:
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