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
    
    # Parameters from Pine Script
    PP = 5  # Pivot Period
    ATR_PERIOD = 55
    
    # Initialize columns
    high = df['high']
    low = df['low']
    close = df['close']
    open_price = df['open']
    
    # Calculate pivothigh and pivotlow
    def pivothigh(series, left_len, right_len):
        result = pd.Series(False, index=series.index)
        for i in range(right_len, len(series) - left_len):
            if all(series.iloc[i] >= series.iloc[i - left_len:i]) and all(series.iloc[i] >= series.iloc[i + 1:i + right_len + 1]):
                result.iloc[i] = True
        return result
    
    def pivotlow(series, left_len, right_len):
        result = pd.Series(False, index=series.index)
        for i in range(right_len, len(series) - left_len):
            if all(series.iloc[i] <= series.iloc[i - left_len:i]) and all(series.iloc[i] <= series.iloc[i + 1:i + right_len + 1]):
                result.iloc[i] = True
        return result
    
    HighPivot = pivothigh(high, PP, PP)
    LowPivot = pivotlow(low, PP, PP)
    
    # Calculate ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    ATR = tr.ewm(alpha=1/ATR_PERIOD, adjust=False).mean()
    
    # ZigZag logic arrays
    zigzag_type = []
    zigzag_value = []
    zigzag_index = []
    
    # Track last pivot information
    last_pivot_type = None
    last_pivot_index = 0
    last_pivot_value = 0
    
    # BoS and ChoCh tracking
    Bullish_Major_BoS = False
    Bearish_Major_BoS = False
    Bullish_Major_ChoCh = False
    Bearish_Major_ChoCh = False
    
    # Trade flags
    dbTradeTriggered = False  # Bearish trade
    dtTradeTriggered = False  # Bullish trade
    isLongOpen = False
    isShortOpen = False
    
    # Fib level for entry (0.782)
    FIB_LEVEL = 0.782
    
    entries = []
    trade_num = 1
    
    # Process each bar
    for i in range(1, len(df)):
        ts = df['time'].iloc[i]
        
        # Update ZigZag on pivot detection
        if HighPivot.iloc[i] or LowPivot.iloc[i]:
            if len(zigzag_type) == 0:
                # First pivot
                if HighPivot.iloc[i]:
                    zigzag_type.append('H')
                    zigzag_value.append(high.iloc[i])
                    zigzag_index.append(i)
                    last_pivot_type = 'H'
                    last_pivot_value = high.iloc[i]
                    last_pivot_index = i
                elif LowPivot.iloc[i]:
                    zigzag_type.append('L')
                    zigzag_value.append(low.iloc[i])
                    zigzag_index.append(i)
                    last_pivot_type = 'L'
                    last_pivot_value = low.iloc[i]
                    last_pivot_index = i
            else:
                last_type = zigzag_type[-1]
                last_val = zigzag_value[-1]
                
                if last_type in ['L', 'LL']:
                    if low.iloc[i] < last_val:
                        # Lower low - update
                        zigzag_type[-1] = 'LL' if len(zigzag_type) > 2 and zigzag_value[-3] > low.iloc[i] else 'L'
                        zigzag_value[-1] = low.iloc[i]
                        zigzag_index[-1] = i
                        last_pivot_type = zigzag_type[-1]
                        last_pivot_value = low.iloc[i]
                        last_pivot_index = i
                    else:
                        # Higher high
                        new_type = 'HH' if len(zigzag_type) < 2 or zigzag_value[-2] < high.iloc[i] else 'LH'
                        zigzag_type.append(new_type)
                        zigzag_value.append(high.iloc[i])
                        zigzag_index.append(i)
                        last_pivot_type = new_type
                        last_pivot_value = high.iloc[i]
                        last_pivot_index = i
                elif last_type in ['H', 'HH']:
                    if high.iloc[i] > last_val:
                        # Higher high - update
                        zigzag_type[-1] = 'HH' if len(zigzag_type) > 2 and zigzag_value[-3] < high.iloc[i] else 'H'
                        zigzag_value[-1] = high.iloc[i]
                        zigzag_index[-1] = i
                        last_pivot_type = zigzag_type[-1]
                        last_pivot_value = high.iloc[i]
                        last_pivot_index = i
                    else:
                        # Lower low
                        new_type = 'LL' if len(zigzag_type) < 2 or zigzag_value[-2] > low.iloc[i] else 'HL'
                        zigzag_type.append(new_type)
                        zigzag_value.append(low.iloc[i])
                        zigzag_index.append(i)
                        last_pivot_type = new_type
                        last_pivot_value = low.iloc[i]
                        last_pivot_index = i
        
        # Detect Major Market Structure when we have enough pivots
        if len(zigzag_type) >= 3:
            # Check for BoS (Break of Structure)
            if zigzag_type[-3] in ['HH', 'LH'] and zigzag_type[-2] in ['HL', 'LL'] and zigzag_type[-1] in ['LH', 'HH']:
                Bearish_Major_BoS = True
                Bullish_Major_BoS = False
            elif zigzag_type[-3] in ['LL', 'HL'] and zigzag_type[-2] in ['LH', 'HH'] and zigzag_type[-1] in ['HL', 'LL']:
                Bullish_Major_BoS = True
                Bearish_Major_BoS = False
            
            # Check for ChoCh (Change of Character)
            if zigzag_type[-2] == 'HH' and zigzag_type[-1] == 'HL':
                Bearish_Major_ChoCh = True
            elif zigzag_type[-2] == 'LL' and zigzag_type[-1] == 'LH':
                Bullish_Major_ChoCh = True
        
        # Entry Logic based on patterns
        if len(zigzag_type) >= 4:
            # Get recent pivot values for Fibonacci calculation
            recent_highs = [zigzag_value[j] for j in range(len(zigzag_type)) if zigzag_type[j] in ['H', 'HH']]
            recent_lows = [zigzag_value[j] for j in range(len(zigzag_type)) if zigzag_type[j] in ['L', 'LL']]
            
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                last_high = max(recent_highs[-2:])
                last_low = min(recent_lows[-2:])
                range_val = last_high - last_low
                fib_782 = last_low + range_val * FIB_LEVEL
                
                # Bearish Entry (Short): After bearish structure, price approaches fib level from above
                if Bearish_Major_BoS or Bearish_Major_ChoCh:
                    if close.iloc[i] <= fib_782 and close.iloc[i-1] > fib_782 and not isShortOpen:
                        dbTradeTriggered = True
                
                # Bullish Entry (Long): After bullish structure, price approaches fib level from below
                if Bullish_Major_BoS or Bullish_Major_ChoCh:
                    if close.iloc[i] >= fib_782 and close.iloc[i-1] < fib_782 and not isLongOpen:
                        dtTradeTriggered = True
                
                # Check for engulfing pattern near fib level
                current_body = abs(close.iloc[i] - open_price.iloc[i])
                prev_body = abs(close.iloc[i-1] - open_price.iloc[i-1])
                
                # Bullish engulfing
                if open_price.iloc[i] < close.iloc[i-1] and close.iloc[i] > open_price.iloc[i-1]:
                    if Bullish_Major_BoS and dtTradeTriggered and not isLongOpen:
                        # Create long entry
                        entry_price = close.iloc[i]
                        entries.append({
                            'trade_num': trade_num,
                            'direction': 'long',
                            'entry_ts': int(ts),
                            'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                            'entry_price_guess': float(entry_price),
                            'exit_ts': 0,
                            'exit_time': '',
                            'exit_price_guess': 0.0,
                            'raw_price_a': float(entry_price),
                            'raw_price_b': float(entry_price)
                        })
                        isLongOpen = True
                        dtTradeTriggered = False
                        trade_num += 1
                
                # Bearish engulfing
                if open_price.iloc[i] > close.iloc[i-1] and close.iloc[i] < open_price.iloc[i-1]:
                    if Bearish_Major_BoS and dbTradeTriggered and not isShortOpen:
                        # Create short entry
                        entry_price = close.iloc[i]
                        entries.append({
                            'trade_num': trade_num,
                            'direction': 'short',
                            'entry_ts': int(ts),
                            'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                            'entry_price_guess': float(entry_price),
                            'exit_ts': 0,
                            'exit_time': '',
                            'exit_price_guess': 0.0,
                            'raw_price_a': float(entry_price),
                            'raw_price_b': float(entry_price)
                        })
                        isShortOpen = True
                        dbTradeTriggered = False
                        trade_num += 1
                
                # Reset flags when positions are closed (simplified logic)
                if isLongOpen and close.iloc[i] < last_low:
                    isLongOpen = False
                if isShortOpen and close.iloc[i] > last_high:
                    isShortOpen = False
        
        # Alternative entry: Price rejection at major/minor levels
        if len(zigzag_type) >= 2:
            # Detect wicks (rejections) at recent pivots
            if last_pivot_type in ['H', 'HH']:
                pivot_price = last_pivot_value
                # Bearish rejection: price touched pivot but closed below
                if high.iloc[i] >= pivot_price * 0.999 and close.iloc[i] < pivot_price * 0.99:
                    if Bearish_Major_BoS and not isShortOpen:
                        entry_price = close.iloc[i]
                        entries.append({
                            'trade_num': trade_num,
                            'direction': 'short',
                            'entry_ts': int(ts),
                            'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                            'entry_price_guess': float(entry_price),
                            'exit_ts': 0,
                            'exit_time': '',
                            'exit_price_guess': 0.0,
                            'raw_price_a': float(entry_price),
                            'raw_price_b': float(entry_price)
                        })
                        isShortOpen = True
                        trade_num += 1
            
            if last_pivot_type in ['L', 'LL']:
                pivot_price = last_pivot_value
                # Bullish rejection: price touched pivot but closed above
                if low.iloc[i] <= pivot_price * 1.001 and close.iloc[i] > pivot_price * 1.01:
                    if Bullish_Major_BoS and not isLongOpen:
                        entry_price = close.iloc[i]
                        entries.append({
                            'trade_num': trade_num,
                            'direction': 'long',
                            'entry_ts': int(ts),
                            'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                            'entry_price_guess': float(entry_price),
                            'exit_ts': 0,
                            'exit_time': '',
                            'exit_price_guess': 0.0,
                            'raw_price_a': float(entry_price),
                            'raw_price_b': float(entry_price)
                        })
                        isLongOpen = True
                        trade_num += 1
    
    return entries