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
    
    # Wilder RSI implementation
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Wilder ATR implementation
    def wilder_atr(df, length):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        return atr
    
    # Parameters from Pine Script
    PP = 5
    atrMultiplier = 2.5
    riskPerTrade = 1.0
    
    # Calculate indicators
    df = df.copy()
    df['atr55'] = wilder_atr(df, 55)
    df['atr14'] = wilder_atr(df, 14)
    df['volume_sma9'] = df['volume'].rolling(9).mean()
    df['volfilt'] = df['volume'].shift(1) > df['volume_sma9'] * 1.5
    
    # Pivot calculations
    df['HighPivot'] = df['high'].rolling(window=PP+1).max().shift(1) == df['high']
    df['LowPivot'] = df['low'].rolling(window=PP+1).min().shift(1) == df['low']
    
    # Initialize structure tracking variables
    n = len(df)
    ArrayType = []
    ArrayValue = []
    ArrayIndex = []
    
    Major_HighLevel = np.nan
    Major_LowLevel = np.nan
    Major_HighIndex = np.nan
    Major_LowIndex = np.nan
    Major_HighType = ''
    Major_LowType = ''
    
    lowestlow = np.nan
    Highesthigh = np.nan
    
    Bullish_Major_ChoCh = False
    Bearish_Major_ChoCh = False
    Bullish_Major_BoS = False
    Bearish_Major_BoS = False
    
    # Entry conditions
    entry_long_cond = pd.Series(False, index=df.index)
    entry_short_cond = pd.Series(False, index=df.index)
    
    for i in range(PP * 2 + 5, n):
        high_val = df['high'].iloc[i]
        low_val = df['low'].iloc[i]
        close_val = df['close'].iloc[i]
        high_index = i
        low_index = i
        bar_index = i
        
        # Check for pivots at this bar
        high_pivot = df['high'].iloc[i] >= df['high'].iloc[max(0, i-PP):i+1].max() if i >= PP else False
        low_pivot = df['low'].iloc[i] <= df['low'].iloc[max(0, i-PP):i+1].min() if i >= PP else False
        
        # Update lowestlow and Highesthigh
        if not np.isnan(Major_LowLevel) and low_val < Major_LowLevel:
            lowestlow = low_val
        elif not np.isnan(Major_LowLevel):
            lowestlow = Major_LowLevel
        else:
            lowestlow = low_val
            
        if not np.isnan(Major_HighLevel) and high_val > Major_HighLevel:
            Highesthigh = high_val
        elif not np.isnan(Major_HighLevel):
            Highesthigh = Major_HighLevel
        else:
            Highesthigh = high_val
        
        # Array management logic (simplified structure detection)
        if high_pivot or low_pivot:
            if len(ArrayType) == 0:
                if high_pivot:
                    ArrayType.append('H')
                    ArrayValue.append(high_val)
                    ArrayIndex.append(high_index)
                    Major_HighLevel = high_val
                    Major_HighIndex = high_index
                    Major_HighType = 'H'
                else:
                    ArrayType.append('L')
                    ArrayValue.append(low_val)
                    ArrayIndex.append(low_index)
                    Major_LowLevel = low_val
                    Major_LowIndex = low_index
                    Major_LowType = 'L'
            else:
                last_type = ArrayType[-1]
                last_val = ArrayValue[-1]
                
                if high_pivot:
                    if last_type in ['L', 'LL']:
                        if low_val < last_val:
                            ArrayType.pop()
                            ArrayValue.pop()
                            ArrayIndex.pop()
                            new_type = 'LL' if (len(ArrayType) > 1 and ArrayValue[-1] < low_val) else 'L'
                            ArrayType.append(new_type)
                            ArrayValue.append(low_val)
                            ArrayIndex.append(low_index)
                            Major_LowLevel = low_val
                            Major_LowIndex = low_index
                            Major_LowType = new_type
                        else:
                            new_type = 'LH' if (len(ArrayType) > 1 and ArrayValue[-1] < high_val) else 'H'
                            ArrayType.append(new_type)
                            ArrayValue.append(high_val)
                            ArrayIndex.append(high_index)
                            Major_HighLevel = high_val
                            Major_HighIndex = high_index
                            Major_HighType = new_type
                    elif last_type in ['H', 'HH']:
                        if high_val > last_val:
                            ArrayType.pop()
                            ArrayValue.pop()
                            ArrayIndex.pop()
                            new_type = 'HH' if (len(ArrayType) > 1 and ArrayValue[-1] < high_val) else 'H'
                            ArrayType.append(new_type)
                            ArrayValue.append(high_val)
                            ArrayIndex.append(high_index)
                            Major_HighLevel = high_val
                            Major_HighIndex = high_index
                            Major_HighType = new_type
                    elif last_type == 'LH':
                        if high_val < last_val:
                            new_type = 'LL' if (len(ArrayType) > 1 and ArrayValue[-1] < low_val) else 'L'
                            ArrayType.append(new_type)
                            ArrayValue.append(low_val)
                            ArrayIndex.append(low_index)
                            Major_LowLevel = low_val
                            Major_LowIndex = low_index
                            Major_LowType = new_type
                        elif high_val > last_val:
                            ArrayType.pop()
                            ArrayValue.pop()
                            ArrayIndex.pop()
                            if close_val < last_val:
                                new_type = 'HH' if (len(ArrayType) > 1 and ArrayValue[-1] < high_val) else 'H'
                                ArrayType.append(new_type)
                                ArrayValue.append(high_val)
                                ArrayIndex.append(high_index)
                                Major_HighLevel = high_val
                                Major_HighIndex = high_index
                                Major_HighType = new_type
                            else:
                                new_type = 'LL' if (len(ArrayType) > 1 and ArrayValue[-1] < low_val) else 'L'
                                ArrayType.append(new_type)
                                ArrayValue.append(low_val)
                                ArrayIndex.append(low_index)
                                Major_LowLevel = low_val
                                Major_LowIndex = low_index
                                Major_LowType = new_type
                    elif last_type == 'HL':
                        if low_val > last_val:
                            new_type = 'LH' if (len(ArrayType) > 1 and ArrayValue[-1] < high_val) else 'H'
                            ArrayType.append(new_type)
                            ArrayValue.append(high_val)
                            ArrayIndex.append(high_index)
                            Major_HighLevel = high_val
                            Major_HighIndex = high_index
                            Major_HighType = new_type
                        elif low_val < last_val:
                            if close_val > last_val:
                                ArrayType.pop()
                                ArrayValue.pop()
                                ArrayIndex.pop()
                                new_type = 'LL' if (len(ArrayType) > 1 and ArrayValue[-1] < low_val) else 'L'
                                ArrayType.append(new_type)
                                ArrayValue.append(low_val)
                                ArrayIndex.append(low_index)
                                Major_LowLevel = low_val
                                Major_LowIndex = low_index
                                Major_LowType = new_type
                            else:
                                new_type = 'LH' if (len(ArrayType) > 1 and ArrayValue[-1] < high_val) else 'H'
                                ArrayType.append(new_type)
                                ArrayValue.append(high_val)
                                ArrayIndex.append(high_index)
                                Major_HighLevel = high_val
                                Major_HighIndex = high_index
                                Major_HighType = new_type
                                
                elif low_pivot:
                    if last_type in ['H', 'HH']:
                        if high_val > last_val:
                            new_type = 'HH' if (len(ArrayType) > 1 and ArrayValue[-1] < high_val) else 'H'
                            ArrayType.append(new_type)
                            ArrayValue.append(high_val)
                            ArrayIndex.append(high_index)
                            Major_HighLevel = high_val
                            Major_HighIndex = high_index
                            Major_HighType = new_type
                        else:
                            new_type = 'HL' if (len(ArrayType) > 1 and ArrayValue[-1] < low_val) else 'L'
                            ArrayType.append(new_type)
                            ArrayValue.append(low_val)
                            ArrayIndex.append(low_index)
                            Major_LowLevel = low_val
                            Major_LowIndex = low_index
                            Major_LowType = new_type
                    elif last_type in ['L', 'LL']:
                        if low_val < last_val:
                            ArrayType.pop()
                            ArrayValue.pop()
                            ArrayIndex.pop()
                            new_type = 'LL' if (len(ArrayType) > 1 and ArrayValue[-1] < low_val) else 'L'
                            ArrayType.append(new_type)
                            ArrayValue.append(low_val)
                            ArrayIndex.append(low_index)
                            Major_LowLevel = low_val
                            Major_LowIndex = low_index
                            Major_LowType = new_type
                    elif last_type == 'LH':
                        if low_val < last_val:
                            new_type = 'LL' if (len(ArrayType) > 1 and ArrayValue[-1] < low_val) else 'L'
                            ArrayType.append(new_type)
                            ArrayValue.append(low_val)
                            ArrayIndex.append(low_index)
                            Major_LowLevel = low_val
                            Major_LowIndex = low_index
                            Major_LowType = new_type
                    elif last_type == 'HL':
                        if low_val > last_val:
                            new_type = 'LH' if (len(ArrayType) > 1 and ArrayValue[-1] < high_val) else 'H'
                            ArrayType.append(new_type)
                            ArrayValue.append(high_val)
                            ArrayIndex.append(high_index)
                            Major_HighLevel = high_val
                            Major_HighIndex = high_index
                            Major_HighType = new_type
                        elif low_val < last_val:
                            if close_val > last_val:
                                new_type = 'LL' if (len(ArrayType) > 1 and ArrayValue[-1] < low_val) else 'L'
                                ArrayType.append(new_type)
                                ArrayValue.append(low_val)
                                ArrayIndex.append(low_index)
                                Major_LowLevel = low_val
                                Major_LowIndex = low_index
                                Major_LowType = new_type
                            else:
                                ArrayType.pop()
                                ArrayValue.pop()
                                ArrayIndex.pop()
                                new_type = 'LH' if (len(ArrayType) > 1 and ArrayValue[-1] < high_val) else 'H'
                                ArrayType.append(new_type)
                                ArrayValue.append(high_val)
                                ArrayIndex.append(high_index)
                                Major_HighLevel = high_val
                                Major_HighIndex = high_index
                                Major_HighType = new_type
        
        # Entry logic based on structure detection
        if len(ArrayType) >= 3 and i > PP * 2 + 10:
            # Detect BoS and ChoCh patterns
            current_type = ArrayType[-1]
            prev_type = ArrayType[-2]
            prev2_type = ArrayType[-3] if len(ArrayType) > 2 else ''
            
            # Major Bullish BoS: Higher high after higher low (uptrend continuation)
            if len(ArrayType) >= 2:
                if current_type == 'HH' and prev_type in ['L', 'LL'] and ArrayValue[-1] > ArrayValue[-2]:
                    if not Bullish_Major_BoS:
                        Bullish_Major_BoS = True
                        # Check volfilt and time
                        if df['volfilt'].iloc[i]:
                            ts = df['time'].iloc[i]
                            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                            if 14 <= dt.hour < 18:
                                entry_long_cond.iloc[i] = True
                                Bullish_Major_BoS = False  # Reset after signal
                
                # Major Bearish BoS: Lower low after lower high (downtrend continuation)
                elif current_type == 'LL' and prev_type in ['H', 'HH'] and ArrayValue[-1] < ArrayValue[-2]:
                    if not Bearish_Major_BoS:
                        Bearish_Major_BoS = True
                        if df['volfilt'].iloc[i]:
                            ts = df['time'].iloc[i]
                            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                            if 14 <= dt.hour < 18:
                                entry_short_cond.iloc[i] = True
                                Bearish_Major_BoS = False
                
                # Bullish ChoCh: Structure change from HH to lower high (potential reversal up)
                if current_type == 'LH' and prev_type in ['HH', 'H']:
                    if not Bullish_Major_ChoCh:
                        Bullish_Major_ChoCh = True
                        if df['volfilt'].iloc[i]:
                            ts = df['time'].iloc[i]
                            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                            if 14 <= dt.hour < 18:
                                entry_long_cond.iloc[i] = True
                                Bullish_Major_ChoCh = False
                
                # Bearish ChoCh: Structure change from LL to higher low (potential reversal down)
                elif current_type == 'HL' and prev_type in ['LL', 'L']:
                    if not Bearish_Major_ChoCh:
                        Bearish_Major_ChoCh = True
                        if df['volfilt'].iloc[i]:
                            ts = df['time'].iloc[i]
                            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                            if 14 <= dt.hour < 18:
                                entry_short_cond.iloc[i] = True
                                Bearish_Major_ChoCh = False
    
    # Build entry list
    entries = []
    trade_num = 1
    
    # Long entries
    long_indices = entry_long_cond[entry_long_cond].index.tolist()
    for idx in long_indices:
        ts = int(df['time'].iloc[idx])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = float(df['close'].iloc[idx])
        entries.append({
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
    
    # Short entries
    short_indices = entry_short_cond[entry_short_cond].index.tolist()
    for idx in short_indices:
        ts = int(df['time'].iloc[idx])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = float(df['close'].iloc[idx])
        entries.append({
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
    
    return entries