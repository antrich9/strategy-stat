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
    
    # Make copies to avoid modifying original df
    open_s = df['open'].copy()
    high_s = df['high'].copy()
    low_s = df['low'].copy()
    close_s = df['close'].copy()
    volume_s = df['volume'].copy()
    time_s = df['time'].copy()
    
    # Define London time windows (7:45-9:45 and 14:45-16:45)
    # We'll filter by hour (UTC equivalent to Europe/London in this context)
    hours = pd.to_datetime(time_s, unit='s', utc=True).dt.hour
    minutes = pd.to_datetime(time_s, unit='s', utc=True).dt.minute
    
    # Morning window: 7:45 to 9:45
    in_morning = ((hours == 7) & (minutes >= 45)) | ((hours == 8)) | ((hours == 9) & (minutes <= 45))
    # Afternoon window: 14:45 to 16:45
    in_afternoon = ((hours == 14) & (minutes >= 45)) | ((hours == 15)) | ((hours == 16) & (minutes <= 45))
    is_within_time_window = in_morning | in_afternoon
    
    # Volume filter: volume[1] > ta.sma(volume, 9) * 1.5
    vol_sma9 = volume_s.rolling(9).mean()
    vol_filt = volume_s.shift(1) > vol_sma9 * 1.5
    
    # ATR filter using Wilder's method: ta.atr(20) / 1.5
    high_low = high_s - low_s
    high_close_prev = (high_s - close_s.shift(1)).abs()
    low_close_prev = (low_s - close_s.shift(1)).abs()
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    # Wilder's ATR
    atr = true_range.ewm(alpha=1/20, adjust=False).mean() / 1.5
    
    # ATR filter: (low - high[2] > atr) or (low[2] - high > atr)
    atr_filt = ((low_s - high_s.shift(2) > atr) | (low_s.shift(2) - high_s > atr))
    
    # Trend filter using SMA
    loc = close_s.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    
    # Bullish FVG: low > high[2] and volfilt and atrfilt and locfiltb
    bfvg = (low_s > high_s.shift(2)) & vol_filt & atr_filt & loc2
    
    # Bearish FVG: high < low[2] and volfilt and atrfilt and locfilts
    sfvg = (high_s < low_s.shift(2)) & vol_filt & atr_filt & (~loc2)
    
    # Order Block and FVG stacked conditions
    # isUp(index): close[index] > open[index]
    # isDown(index): close[index] < open[index]
    # isObUp(index): isDown(index + 1) and isUp(index) and close[index] > high[index + 1]
    # isObDown(index): isUp(index + 1) and isDown(index) and close[index] < low[index + 1]
    # isFvgUp(index): low[index] > high[index + 2]
    # isFvgDown(index): high[index] < low[index + 2]
    
    # Current state (bar 0)
    is_up_0 = close_s > open_s
    is_down_0 = close_s < open_s
    is_fvg_up_0 = low_s > high_s.shift(2)
    is_fvg_down_0 = high_s < low_s.shift(2)
    
    # Bar 1
    is_up_1 = close_s.shift(1) > open_s.shift(1)
    is_down_1 = close_s.shift(1) < open_s.shift(1)
    
    # Stacked conditions for bar 1
    is_ob_up_1 = is_down_1.shift(1) & is_up_1 & (close_s.shift(1) > high_s.shift(2))
    is_ob_down_1 = is_up_1.shift(1) & is_down_1 & (close_s.shift(1) < low_s.shift(2))
    
    # Combine OB and FVG for stacked signal
    stacked_bullish = is_ob_up_1 & is_fvg_up_0
    stacked_bearish = is_ob_down_1 & is_fvg_down_0
    
    # Entry conditions: within time window AND stacked FVG AND FVG signal
    long_condition = is_within_time_window & stacked_bullish & bfvg
    short_condition = is_within_time_window & stacked_bearish & sfvg
    
    # Iterate through bars
    for i in range(len(df)):
        # Skip if any required indicator is NaN
        if i < 10:  # Need enough bars for rolling calculations
            continue
        if pd.isna(atr.iloc[i]) or pd.isna(loc.iloc[i]):
            continue
        if pd.isna(vol_filt.iloc[i]):
            continue
        if pd.isna(stacked_bullish.iloc[i]) or pd.isna(stacked_bearish.iloc[i]):
            continue
        if pd.isna(bfvg.iloc[i]) or pd.isna(sfvg.iloc[i]):
            continue
        if pd.isna(is_within_time_window.iloc[i]):
            continue
            
        if long_condition.iloc[i]:
            ts = int(time_s.iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(close_s.iloc[i])
            
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            
        if short_condition.iloc[i]:
            ts = int(time_s.iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(close_s.iloc[i])
            
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
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