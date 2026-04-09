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
    
    # Input defaults from Pine Script
    inp1 = False  # Volume Filter
    inp2 = False  # ATR Filter
    inp3 = False  # Trend Filter
    
    # Trading windows from inputs
    start_hour_1 = 7
    end_hour_1 = 10
    end_minute_1 = 59
    start_hour_2 = 15
    end_hour_2 = 16
    end_minute_2 = 59
    
    # Copy df to avoid modifying original
    data = df.copy()
    
    # Fill NaN values with forward fill then backward fill for complete data
    data['high'] = data['high'].fillna(method='ffill').fillna(method='bfill')
    data['low'] = data['low'].fillna(method='ffill').fillna(method='bfill')
    data['close'] = data['close'].fillna(method='ffill').fillna(method='bfill')
    data['open'] = data['open'].fillna(method='ffill').fillna(method='bfill')
    data['volume'] = data['volume'].fillna(0)
    
    n = len(data)
    if n < 3:
        return []
    
    # Calculate indicators
    # Volume filter
    if inp1:
        sma_vol = data['volume'].rolling(9).mean()
        volfilt = data['volume'].shift(1) > sma_vol * 1.5
    else:
        volfilt = pd.Series(True, index=data.index)
    
    # ATR using Wilder's method
    high = data['high']
    low = data['low']
    close = data['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = pd.Series(index=data.index, dtype=float)
    atr.iloc[19] = tr.iloc[:20].mean()
    for i in range(20, n):
        atr.iloc[i] = (atr.iloc[i-1] * 19 + tr.iloc[i]) / 20
    atr = atr / 1.5
    
    # ATR filter
    if inp2:
        atrfilt = ((low - high.shift(2) > atr) | (low.shift(2) - high > atr))
    else:
        atrfilt = pd.Series(True, index=data.index)
    
    # Trend filter
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2 if inp3 else pd.Series(True, index=data.index)
    locfilts = ~loc2 if inp3 else pd.Series(True, index=data.index)
    
    # FVG conditions
    bfvg = (low > high.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (high < low.shift(2)) & volfilt & atrfilt & locfilts
    
    # Trading window logic
    def get_dst_offset(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        month = dt.month
        day = dt.day
        dayofweek = dt.weekday()
        
        # DST check: Last Sunday March to Last Sunday October
        # Find last Sunday of March
        march_31 = datetime(dt.year, 3, 31, tzinfo=timezone.utc)
        last_sunday_march = 31 - (march_31.weekday() + 1) % 7
        dst_start = datetime(dt.year, 3, last_sunday_march, tzinfo=timezone.utc)
        
        oct_31 = datetime(dt.year, 10, 31, tzinfo=timezone.utc)
        last_sunday_oct = 31 - (oct_31.weekday() + 1) % 7
        dst_end = datetime(dt.year, 10, last_sunday_oct, tzinfo=timezone.utc)
        
        if dst_start <= dt < dst_end:
            return 3600  # DST offset in seconds
        return 0
    
    in_trading_window = pd.Series(False, index=data.index)
    
    for i in range(n):
        ts = data['time'].iloc[i]
        offset = get_dst_offset(ts)
        adjusted_ts = ts + offset
        dt = datetime.fromtimestamp(adjusted_ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        
        # Window 1: 07:00 - 10:59
        in_window_1 = (hour >= start_hour_1 and hour <= end_hour_1) and not (hour == end_hour_1 and minute > end_minute_1)
        # Window 2: 15:00 - 16:59
        in_window_2 = (hour >= start_hour_2 and hour <= end_hour_2) and not (hour == end_hour_2 and minute > end_minute_2)
        
        in_trading_window.iloc[i] = in_window_1 or in_window_2
    
    # OB/FVG stacking conditions (for plotting, not entries based on code structure)
    # These are evaluated at bar i using lookback
    def is_up(idx):
        return close.iloc[idx] > data['open'].iloc[idx]
    
    def is_down(idx):
        return close.iloc[idx] < data['open'].iloc[idx]
    
    def is_ob_up(idx):
        return is_down(idx + 1) and is_up(idx) and close.iloc[idx] > high.iloc[idx + 1]
    
    def is_ob_down(idx):
        return is_up(idx + 1) and is_down(idx) and close.iloc[idx] < low.iloc[idx + 1]
    
    def is_fvg_up(idx):
        return low.iloc[idx] > high.iloc[idx + 2]
    
    def is_fvg_down(idx):
        return high.iloc[idx] < low.iloc[idx + 2]
    
    # Entries based on FVG conditions within trading window
    entries = []
    trade_num = 1
    
    for i in range(2, n):
        if in_trading_window.iloc[i]:
            # Long entry on bullish FVG
            if bfvg.iloc[i]:
                ts = int(data['time'].iloc[i])
                entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': entry_time,
                    'entry_price_guess': float(close.iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(close.iloc[i]),
                    'raw_price_b': float(close.iloc[i])
                })
                trade_num += 1
            
            # Short entry on bearish FVG
            if sfvg.iloc[i]:
                ts = int(data['time'].iloc[i])
                entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': entry_time,
                    'entry_price_guess': float(close.iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(close.iloc[i]),
                    'raw_price_b': float(close.iloc[i])
                })
                trade_num += 1
    
    return entries