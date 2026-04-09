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
    
    open_series = df['open']
    high_series = df['high']
    low_series = df['low']
    close_series = df['close']
    volume_series = df['volume']
    time_series = df['time']
    
    # Number of bars
    n = len(df)
    
    # Helper functions for OB and FVG
    def is_up(idx):
        return close_series.iloc[idx] > open_series.iloc[idx]
    
    def is_down(idx):
        return close_series.iloc[idx] < open_series.iloc[idx]
    
    def is_ob_up(idx):
        return is_down(idx + 1) and is_up(idx) and close_series.iloc[idx] > high_series.iloc[idx + 1]
    
    def is_ob_down(idx):
        return is_up(idx + 1) and is_down(idx) and close_series.iloc[idx] < low_series.iloc[idx + 1]
    
    def is_fvg_up(idx):
        return low_series.iloc[idx] > high_series.iloc[idx + 2]
    
    def is_fvg_down(idx):
        return high_series.iloc[idx] < low_series.iloc[idx + 2]
    
    # Initialize OB and FVG series
    ob_up = pd.Series(False, index=df.index)
    ob_down = pd.Series(False, index=df.index)
    fvg_up = pd.Series(False, index=df.index)
    fvg_down = pd.Series(False, index=df.index)
    
    # Calculate OB and FVG for indices where valid
    for i in range(2, n - 2):
        try:
            ob_up.iloc[i] = is_ob_up(i)
            ob_down.iloc[i] = is_ob_down(i)
            fvg_up.iloc[i] = is_fvg_up(i)
            fvg_down.iloc[i] = is_fvg_down(i)
        except:
            pass
    
    # Wilder ATR calculation
    def wilder_atr(period):
        tr = pd.Series(0.0, index=df.index)
        for i in range(1, n):
            high_val = high_series.iloc[i]
            low_val = low_series.iloc[i]
            prev_close = close_series.iloc[i - 1]
            tr.iloc[i] = max(high_val - low_val, abs(high_val - prev_close), abs(low_val - prev_close))
        
        atr = pd.Series(np.nan, index=df.index)
        atr.iloc[period] = tr.iloc[1:period + 1].sum()
        
        for i in range(period + 1, n):
            atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + tr.iloc[i]) / period
        
        return atr
    
    # Wilder RSI calculation
    def wilder_rsi(src, period):
        delta = src.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = pd.Series(np.nan, index=df.index)
        avg_loss = pd.Series(np.nan, index=df.index)
        
        avg_gain.iloc[period] = gain.iloc[1:period + 1].mean()
        avg_loss.iloc[period] = loss.iloc[1:period + 1].mean()
        
        for i in range(period + 1, n):
            avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate SMA for volume
    vol_sma_9 = volume_series.rolling(9).mean()
    
    # Calculate ATR
    atr_20 = wilder_atr(20) / 1.5
    
    # Calculate SMA for trend filter
    loc = close_series.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    
    # Volume filter
    volfilt = volume_series.shift(1) > vol_sma_9 * 1.5
    
    # ATR filter
    atrfilt = ((low_series - high_series.shift(2) > atr_20) | (low_series.shift(2) - high_series > atr_20))
    
    # Trend filters
    locfiltb = loc2
    locfilts = ~loc2
    
    # Bullish and Bearish FVG
    bfvg = (low_series > high_series.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (high_series < low_series.shift(2)) & volfilt & atrfilt & locfilts
    
    # Stacked OB + FVG conditions
    stacked_long = ob_up & fvg_up
    stacked_short = ob_down & fvg_down
    
    # PDHL sweep conditions
    prev_day_high = high_series.shift(1).rolling(1440).max()  # Approximate - need daily high
    prev_day_low = low_series.shift(1).rolling(1440).min()   # Approximate - need daily low
    
    # For practical purposes, use rolling high/low with larger window
    pdh_sweep_long = close_series > prev_day_high
    pdl_sweep_short = close_series < prev_day_low
    
    # Trading window (UTC times)
    in_trading_window = pd.Series(False, index=df.index)
    
    for i in df.index:
        ts = time_series.iloc[i]
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        
        # Window 1: 07:00 - 10:59
        in_win1 = (hour >= 7 and hour <= 10) and not (hour == 10 and minute > 59)
        # Window 2: 15:00 - 16:59
        in_win2 = (hour >= 15 and hour <= 16) and not (hour == 16 and minute > 59)
        
        in_trading_window.iloc[i] = in_win1 or in_win2
    
    # Entry conditions
    long_entry = stacked_long & in_trading_window
    short_entry = stacked_short & in_trading_window
    
    # Build result
    entries = []
    trade_num = 1
    
    for i in range(n):
        if pd.isna(close_series.iloc[i]):
            continue
        
        if long_entry.iloc[i] if i < len(long_entry) else False:
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(time_series.iloc[i]),
                'entry_time': datetime.fromtimestamp(time_series.iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close_series.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close_series.iloc[i]),
                'raw_price_b': float(close_series.iloc[i])
            }
            entries.append(entry)
            trade_num += 1
        elif short_entry.iloc[i] if i < len(short_entry) else False:
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(time_series.iloc[i]),
                'entry_time': datetime.fromtimestamp(time_series.iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close_series.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close_series.iloc[i]),
                'raw_price_b': float(close_series.iloc[i])
            }
            entries.append(entry)
            trade_num += 1
    
    return entries