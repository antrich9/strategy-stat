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
    trade_num = 0
    
    # Helper functions for OB and FVG detection
    def is_up(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]
    
    def is_down(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]
    
    def is_ob_up(idx):
        return (is_down(idx + 1) and is_up(idx) and 
                df['close'].iloc[idx] > df['high'].iloc[idx + 1])
    
    def is_ob_down(idx):
        return (is_up(idx + 1) and is_down(idx) and 
                df['close'].iloc[idx] < df['low'].iloc[idx + 1])
    
    def is_fvg_up(idx):
        return df['low'].iloc[idx] > df['high'].iloc[idx + 2]
    
    def is_fvg_down(idx):
        return df['high'].iloc[idx] < df['low'].iloc[idx + 2]
    
    # Calculate SMA for volume filter
    sma9_vol = df['volume'].rolling(9).mean()
    vol_filt = sma9_vol * 1.5
    
    # Calculate ATR (Wilder)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr20 = tr.ewm(span=20, adjust=False).mean()
    atr_thresh = atr20 / 1.5
    
    # Calculate trend filter (SMA 54)
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    
    # London time windows (in UTC)
    # We need to convert df['time'] to datetime to check hour
    times = pd.to_datetime(df['time'], unit='s', utc=True)
    hours = times.dt.hour
    minutes = times.dt.minute
    total_minutes = hours * 60 + minutes
    
    # Morning: 7:45 to 9:45 (465 to 585 minutes)
    # Afternoon: 14:45 to 16:45 (885 to 1005 minutes)
    is_morning_window = (total_minutes >= 465) & (total_minutes < 585)
    is_afternoon_window = (total_minutes >= 885) & (total_minutes < 1005)
    is_within_time_window = is_morning_window | is_afternoon_window
    
    # Calculate FVG conditions
    fvg_bull = df['low'] > df['high'].shift(2)
    fvg_bear = df['high'] < df['low'].shift(2)
    
    # Volume filter
    vol_filt_bool = df['volume'].shift(1) > vol_filt
    
    # ATR filter
    atrfilt_bull = df['low'] - df['high'].shift(2) > atr_thresh
    atrfilt_bear = df['low'].shift(2) - df['high'] > atr_thresh
    
    # Trend filters
    locfiltb = loc2
    locfilts = ~loc2
    
    # Combined FVG signals
    bfvg = fvg_bull & vol_filt_bool & atrfilt_bull & locfiltb
    sfvg = fvg_bear & vol_filt_bool & atrfilt_bear & locfilts
    
    # Detect stacked OB+FVG conditions
    # obUp/obDown are checked at index 1 (previous bar)
    # fvgUp/fvgDown are checked at index 0 (current bar)
    
    # Need at least 3 bars for FVG check
    min_bars = 3
    
    for i in range(min_bars, len(df)):
        if pd.isna(df['high'].iloc[i]) or pd.isna(df['low'].iloc[i]):
            continue
        
        # Skip if not in time window
        if not is_within_time_window.iloc[i]:
            continue
        
        # Skip if indicators have NaN
        if pd.isna(loc.iloc[i]) or pd.isna(locfiltb.iloc[i]) or pd.isna(locfilts.iloc[i]):
            continue
        if pd.isna(atr_thresh.iloc[i]):
            continue
        
        # Check for stacked OB+FVG conditions
        # Long condition: OB bullish on bar i-1 and FVG bullish on bar i
        try:
            ob_up_current = is_ob_up(i - 1)
            ob_down_current = is_ob_down(i - 1)
            fvg_up_current = is_fvg_up(i)
            fvg_down_current = is_fvg_down(i)
        except:
            continue
        
        # Long entry: obUp and fvgUp
        if ob_up_current and fvg_up_current:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
        
        # Short entry: obDown and fvgDown
        if ob_down_current and fvg_down_current:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
    
    return results