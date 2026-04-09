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
    
    entries = []
    trade_num = 1
    
    # Ensure we have enough data
    if len(df) < 50:
        return entries
    
    # Create timezone-aware datetime column
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Convert to London timezone (Europe/London)
    try:
        london_tz = datetime.now(timezone.utc).astimezone().tzname()
        # Use explicit London timezone offset
        from datetime import timedelta
        # For simplicity, use UTC+0 (London in winter) approximation
        df['dt_london'] = df['dt'].dt.tz_convert('Europe/London')
    except:
        # Fallback: assume UTC
        df['dt_london'] = df['dt']
    
    # Extract hour and minute from London time
    df['hour'] = df['dt_london'].dt.hour
    df['minute'] = df['dt_london'].dt.minute
    df['total_minutes'] = df['hour'] * 60 + df['minute']
    
    # Define time windows (in minutes from midnight)
    # Morning: 7:45 to 9:45 (465 to 585 minutes)
    # Afternoon: 14:45 to 16:45 (885 to 1005 minutes)
    morning_start = 7 * 60 + 45  # 465
    morning_end = 9 * 60 + 45    # 585
    afternoon_start = 14 * 60 + 45  # 885
    afternoon_end = 16 * 60 + 45    # 1005
    
    # Time window condition
    is_within_time_window = (
        ((df['total_minutes'] >= morning_start) & (df['total_minutes'] < morning_end)) |
        ((df['total_minutes'] >= afternoon_start) & (df['total_minutes'] < afternoon_end))
    )
    
    # Calculate ATR using Wilder's method
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range calculation
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder ATR(20)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atr_adjusted = atr / 1.5
    
    # Volume filter: volume[1] > sma(volume, 9) * 1.5
    vol_sma = df['volume'].rolling(9).mean()
    vol_filt_base = df['volume'].shift(1) > vol_sma * 1.5
    
    # ATR filter: (low - high[2] > atr) or (low[2] - high > atr)
    atr_filt_base = (low - high.shift(2) > atr_adjusted) | (low.shift(2) - high > atr_adjusted)
    
    # Trend filter: SMA(close, 54) > SMA(close, 54)[1]
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    
    # Bullish conditions
    vol_filt_b = vol_filt_base
    atr_filt_b = atr_filt_base
    loc_filt_b = loc2
    
    # Bearish conditions
    vol_filt_s = vol_filt_base
    atr_filt_s = atr_filt_base
    loc_filt_s = ~loc2
    
    # Bullish FVG: low > high[2] and vol_filt and atr_filt and loc_filt_b
    bfvg = (low > high.shift(2)) & vol_filt_b & atr_filt_b & loc_filt_b
    
    # Bearish FVG: high < low[2] and vol_filt and atr_filt and loc_filt_s
    sfvg = (high < low.shift(2)) & vol_filt_s & atr_filt_s & loc_filt_s
    
    # Entry conditions
    long_condition = bfvg & is_within_time_window
    short_condition = sfvg & is_within_time_window
    
    # Iterate through bars
    for i in range(len(df)):
        if i < 3:
            continue
            
        # Skip if any required indicators are NaN
        if pd.isna(atr_adjusted.iloc[i]) or pd.isna(loc.iloc[i]):
            continue
        
        # Check long entry
        if long_condition.iloc[i]:
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            }
            entries.append(entry)
            trade_num += 1
        
        # Check short entry
        if short_condition.iloc[i]:
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            }
            entries.append(entry)
            trade_num += 1
    
    return entries