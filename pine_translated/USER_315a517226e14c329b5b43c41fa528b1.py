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
    
    # Ensure required columns exist
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        return results
    
    # Copy df to avoid modifying original
    data = df.copy()
    
    # Helper function to check if bar is up (close > open)
    def is_up(idx):
        if idx < 0 or idx >= len(data):
            return False
        return data['close'].iloc[idx] > data['open'].iloc[idx]
    
    # Helper function to check if bar is down (close < open)
    def is_down(idx):
        if idx < 0 or idx >= len(data):
            return False
        return data['close'].iloc[idx] < data['open'].iloc[idx]
    
    # Bullish Order Block: isDown(bar+1) and isUp(bar) and close(bar) > high(bar+1)
    def is_ob_up(idx):
        if idx < 0 or idx + 1 >= len(data):
            return False
        return is_down(idx + 1) and is_up(idx) and data['close'].iloc[idx] > data['high'].iloc[idx + 1]
    
    # Bearish Order Block: isUp(bar+1) and isDown(bar) and close(bar) < low(bar+1)
    def is_ob_down(idx):
        if idx < 0 or idx + 1 >= len(data):
            return False
        return is_up(idx + 1) and is_down(idx) and data['close'].iloc[idx] < data['low'].iloc[idx + 1]
    
    # Bullish FVG: low(bar) > high(bar-2)
    def is_fvg_up(idx):
        if idx < 0 or idx - 2 < 0:
            return False
        return data['low'].iloc[idx] > data['high'].iloc[idx - 2]
    
    # Bearish FVG: high(bar) < low(bar-2)
    def is_fvg_down(idx):
        if idx < 0 or idx - 2 < 0:
            return False
        return data['high'].iloc[idx] < data['low'].iloc[idx - 2]
    
    # Calculate Asia session high and low (01:00 - 05:00 London time)
    # We'll use timestamp-based session detection
    asia_high = np.nan
    asia_low = np.nan
    in_asia_session = False
    in_asia_session_prev = False
    asia_session_highs = []
    asia_session_lows = []
    
    # Process each bar
    for i in range(len(data)):
        current_time = data['time'].iloc[i]
        current_dt = datetime.fromtimestamp(current_time, tz=timezone.utc)
        hour = current_dt.hour
        
        # Check if in Asia session (01:00 - 05:00 UTC)
        in_asia_session = (hour >= 1 and hour < 5)
        
        # Detect new session start
        new_session = in_asia_session and not in_asia_session_prev
        session_end = not in_asia_session and in_asia_session_prev
        
        # At new session, reset high and low
        if new_session:
            asia_high = data['high'].iloc[i]
            asia_low = data['low'].iloc[i]
        
        # During session, update high and low
        if in_asia_session:
            asia_high = max(asia_high, data['high'].iloc[i]) if not np.isnan(asia_high) else data['high'].iloc[i]
            asia_low = min(asia_low, data['low'].iloc[i]) if not np.isnan(asia_low) else data['low'].iloc[i]
        
        # Store session high/low at session end
        if session_end and not np.isnan(asia_high):
            asia_session_highs.append((current_time, asia_high))
            asia_session_lows.append((current_time, asia_low))
        
        in_asia_session_prev = in_asia_session
    
    # For entries, we need to look for sweeps of previous Asia high/low
    # Plus stacked OB and FVG
    
    # Get current bar OB/FVG conditions
    ob_up = is_ob_up(1) if len(data) > 1 else False
    ob_down = is_ob_down(1) if len(data) > 1 else False
    fvg_up = is_fvg_up(0) if len(data) > 0 else False
    fvg_down = is_fvg_down(0) if len(data) > 0 else False
    
    # Volume filter
    vol_sma = data['volume'].rolling(9).mean()
    vol_filt = True  # inp1 = false by default
    
    # ATR filter (using simplified ATR)
    tr = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    atr = tr.rolling(20).mean() / 1.5
    atr_filt = True  # inp2 = false by default
    
    # Trend filter
    loc = data['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    loc_filt_bull = True  # inp3 = false by default
    loc_filt_bear = True  # inp3 = false by default
    
    # Bullish FVG condition
    bfvg = (data['low'] > data['high'].shift(2)) & vol_filt & atr_filt & loc_filt_bull
    
    # Bearish FVG condition
    sfvg = (data['high'] < data['low'].shift(2)) & vol_filt & atr_filt & loc_filt_bear
    
    # PDHL (Previous Day High/Low) sweep conditions
    # Using rolling max/min of high/low from previous day (approximate with 24 periods)
    prev_day_high = data['high'].rolling(24).max().shift(1)
    prev_day_low = data['low'].rolling(24).min().shift(1)
    
    # Entry conditions: Price sweeping PDH/PDL with stacked OB and FVG
    # Bullish: Price sweeps below previous day low, then closes above, with bullish OB and FVG
    # For sweep detection: price crosses above/below key levels
    
    # Check for valid trade time (02-05 and 10-12 UTC)
    valid_trade_time = ((data['time'].dt.hour >= 2) & (data['time'].dt.hour < 5)) | \
                       ((data['time'].dt.hour >= 10) & (data['time'].dt.hour < 12))
    
    # Since we don't have explicit strategy.entry calls, we derive from conditions:
    # Long entry: bfvg and ob_up (bullish FVG and bullish order block stacked)
    # Short entry: sfvg and ob_down (bearish FVG and bearish order block stacked)
    
    long_entry = bfvg & (data['close'] > data['low'].shift(1))  # Price above recent low with bullish FVG
    short_entry = sfvg & (data['close'] < data['high'].shift(1))  # Price below recent high with bearish FVG
    
    # Iterate through bars to generate entries
    for i in range(len(data)):
        # Skip if required lookback bars not available
        if i < 3:
            continue
        
        # Check for NaN in indicators
        if i >= len(bfvg) or pd.isna(bfvg.iloc[i]) if hasattr(bfvg, 'iloc') else False:
            continue
        if i >= len(sfvg) or pd.isna(sfvg.iloc[i]) if hasattr(sfvg, 'iloc') else False:
            continue
        
        # Detect bullish entry
        if long_entry.iloc[i] if hasattr(long_entry, 'iloc') else long_entry[i]:
            trade_num += 1
            entry_ts = int(data['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(data['close'].iloc[i])
            
            results.append({
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
        
        # Detect bearish entry
        if short_entry.iloc[i] if hasattr(short_entry, 'iloc') else short_entry[i]:
            trade_num += 1
            entry_ts = int(data['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(data['close'].iloc[i])
            
            results.append({
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
    
    return results