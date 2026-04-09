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
    # Helper functions for candlestick pattern detection
    def is_up(s):
        return s['close'] > s['open']
    
    def is_down(s):
        return s['close'] < s['open']
    
    # Detect if bar at index is a bullish order block (OB Up)
    # isDown(index + 1) and isUp(index) and close[index] > high[index + 1]
    is_down_prev = is_down(df.shift(1))
    is_up_curr = is_up(df)
    high_prev = df['high'].shift(1)
    ob_up = is_down_prev & is_up_curr & (df['close'] > high_prev)
    
    # Detect if bar at index is a bearish order block (OB Down)
    # isUp(index + 1) and isDown(index) and close[index] < low[index + 1]
    is_up_prev = is_up(df.shift(1))
    is_down_curr = is_down(df)
    low_prev = df['low'].shift(1)
    ob_down = is_up_prev & is_down_curr & (df['close'] < low_prev)
    
    # Detect bullish Fair Value Gap (FVG Up)
    # low[index] > high[index + 2]
    high_2_bars_ago = df['high'].shift(2)
    fvg_up = df['low'] > high_2_bars_ago
    
    # Detect bearish Fair Value Gap (FVG Down)
    # high[index] < low[index + 2]
    low_2_bars_ago = df['low'].shift(2)
    fvg_down = df['high'] < low_2_bars_ago
    
    # Previous Day High (PDH) and Low (PDL) - using rolling 96 bars (24h of 15m bars)
    pdh = df['high'].rolling(96).max().shift(1)
    pdl = df['low'].rolling(96).min().shift(1)
    
    # Detect sweeps of previous day high/low
    pdh_sweep = (df['close'] > pdh) & (df['close'].shift(1) <= pdh)
    pdl_sweep = (df['close'] < pdl) & (df['close'].shift(1) >= pdl)
    
    # Crossover/crossunder detection for PDH/PDL
    pdh_cross = df['close'] > pdh
    pdl_cross = df['close'] < pdl
    
    # Local swing high/low detection (simplified from 4H swing in original)
    swing_high = (
        (df['high'].shift(2) > df['high'].shift(3)) &
        (df['high'].shift(2) >= df['high'].shift(1)) &
        (df['high'].shift(2) >= df['high'].shift(4)) &
        (df['high'].shift(2) >= df['high'].shift(5))
    )
    
    swing_low = (
        (df['low'].shift(2) < df['low'].shift(3)) &
        (df['low'].shift(2) <= df['low'].shift(1)) &
        (df['low'].shift(2) <= df['low'].shift(4)) &
        (df['low'].shift(2) <= df['low'].shift(5))
    )
    
    # Entry conditions: stacked OB/FVG + PDHL sweep
    # Long: OB Up or FVG Up + sweep of PDL (or cross with swing low)
    # Short: OB Down or FVG Down + sweep of PDH (or cross with swing high)
    bullish_entry = (ob_up | fvg_up) & (pdl_sweep | (pdl_cross & swing_low))
    bearish_entry = (ob_down | fvg_down) & (pdh_sweep | (pdh_cross & swing_high))
    
    # Time window filtering (London time: 7:45-9:45 and 14:45-16:45)
    times = pd.to_datetime(df['time'], unit='s', utc=True)
    hours = times.dt.hour
    minutes = times.dt.minute
    weekdays = times.dt.weekday
    
    # Morning window: 7:45 to 9:45
    morning_window = ((hours == 7) & (minutes >= 45)) | ((hours == 8)) | ((hours == 9) & (minutes <= 45))
    # Afternoon window: 14:45 to 16:45
    afternoon_window = ((hours == 14) & (minutes >= 45)) | ((hours == 15)) | ((hours == 16) & (minutes <= 45))
    
    # Exclude Friday mornings (weekday 4 = Friday)
    time_ok = (weekdays != 4) & (morning_window | afternoon_window)
    
    # Apply time filter
    bullish_entry = bullish_entry & time_ok
    bearish_entry = bearish_entry & time_ok
    
    # Remove NaN rows from indicator lookback
    valid_mask = ~(ob_up.isna() | ob_down.isna() | fvg_up.isna() | fvg_down.isna())
    bullish_entry = bullish_entry & valid_mask
    bearish_entry = bearish_entry & valid_mask
    
    # Build entry list
    entries = []
    trade_num = 1
    
    # Long entries
    for i in bullish_entry[bullish_entry].index:
        ts = int(df['time'].iloc[i])
        entries.append({
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
        trade_num += 1
    
    # Short entries
    for i in bearish_entry[bearish_entry].index:
        ts = int(df['time'].iloc[i])
        entries.append({
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
        trade_num += 1
    
    return entries