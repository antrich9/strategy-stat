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
    
    high = df['high']
    low = df['low']
    close = df['close']
    bar_count = len(df)
    
    PP = 5
    atrLength = 55
    
    # ATR calculation (Wilder)
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/atrLength, adjust=False).mean()
    
    # Pivot detection
    pivot_high = np.zeros(bar_count, dtype=bool)
    pivot_low = np.zeros(bar_count, dtype=bool)
    
    for i in range(PP, bar_count - PP):
        is_high = True
        is_low = True
        for j in range(1, PP + 1):
            if high.iloc[i] <= high.iloc[i - j]:
                is_high = False
            if low.iloc[i] >= low.iloc[i + j]:
                is_low = False
        if is_high:
            pivot_high[i] = True
        if is_low:
            pivot_low[i] = True
    
    pivot_high = pd.Series(pivot_high, index=df.index)
    pivot_low = pd.Series(pivot_low, index=df.index)
    
    # ZigZag arrays
    zz_type = []
    zz_value = []
    zz_index = []
    
    for i in range(PP, bar_count - PP):
        if pivot_high.iloc[i] and pivot_low.iloc[i]:
            if len(zz_type) == 0:
                if high.iloc[i] >= low.iloc[i]:
                    zz_type.append('H')
                    zz_value.append(high.iloc[i])
                    zz_index.append(i)
                else:
                    zz_type.append('L')
                    zz_value.append(low.iloc[i])
                    zz_index.append(i)
            else:
                last_type = zz_type[-1]
                if last_type in ['L', 'LL']:
                    if low.iloc[i] < zz_value[-1]:
                        zz_type[-1] = 'LL' if len(zz_type) > 2 and zz_value[-2] < low.iloc[i] else 'L'
                        zz_value[-1] = low.iloc[i]
                        zz_index[-1] = i
                    else:
                        new_type = 'HH' if len(zz_type) < 2 or zz_value[-2] >= high.iloc[i] else 'LH'
                        zz_type.append(new_type)
                        zz_value.append(high.iloc[i])
                        zz_index.append(i)
                else:
                    if high.iloc[i] > zz_value[-1]:
                        zz_type[-1] = 'HH' if len(zz_type) > 2 and zz_value[-2] > high.iloc[i] else 'H'
                        zz_value[-1] = high.iloc[i]
                        zz_index[-1] = i
                    else:
                        new_type = 'LL' if len(zz_type) < 2 or zz_value[-2] <= low.iloc[i] else 'HL'
                        zz_type.append(new_type)
                        zz_value.append(low.iloc[i])
                        zz_index.append(i)
    
    if len(zz_type) < 3:
        return []
    
    # Major and minor structure detection
    major_bull_chosh = np.zeros(bar_count, dtype=bool)
    major_bear_chosh = np.zeros(bar_count, dtype=bool)
    major_bull_bos = np.zeros(bar_count, dtype=bool)
    major_bear_bos = np.zeros(bar_count, dtype=bool)
    minor_bull_chosh = np.zeros(bar_count, dtype=bool)
    minor_bear_chosh = np.zeros(bar_count, dtype=bool)
    minor_bull_bos = np.zeros(bar_count, dtype=bool)
    minor_bear_bos = np.zeros(bar_count, dtype=bool)
    
    def get_last_pivots(arr_type, arr_idx, count):
        types = []
        indices = []
        start = max(0, len(arr_type) - count)
        for k in range(start, len(arr_type)):
            types.append(arr_type[k])
            indices.append(arr_idx[k])
        return types, indices
    
    for i in range(PP, bar_count - PP):
        recent_types, recent_indices = get_last_pivots(zz_type, zz_index, 4)
        if len(recent_indices) < 4:
            continue
        
        # Major structure (swing high/low)
        if len(recent_indices) >= 4:
            types_m = recent_types[-4:]
            idx_m = recent_indices[-4:]
            
            # Major bullish ChoCh
            if types_m[0] == 'H' and types_m[1] == 'L' and types_m[2] == 'H' and types_m[3] == 'L':
                if high[idx_m[2]] > high[idx_m[0]] and low[idx_m[3]] < low[idx_m[1]]:
                    major_bull_chosh[idx_m[3]] = True
            
            # Major bearish ChoCh
            if types_m[0] == 'L' and types_m[1] == 'H' and types_m[2] == 'L' and types_m[3] == 'H':
                if low[idx_m[2]] < low[idx_m[0]] and high[idx_m[3]] > high[idx_m[1]]:
                    major_bear_chosh[idx_m[3]] = True
            
            # Major Bullish BoS
            if types_m[1] == 'L' and types_m[2] == 'H' and types_m[3] == 'L':
                if high[idx_m[2]] > high[idx_m[0]]:
                    major_bull_bos[idx_m[3]] = True
            
            # Major Bearish BoS
            if types_m[1] == 'H' and types_m[2] == 'L' and types_m[3] == 'H':
                if low[idx_m[2]] < low[idx_m[0]]:
                    major_bear_bos[idx_m[3]] = True
        
        # Minor structure (last 2 pivots)
        if len(recent_indices) >= 3:
            types_mn = recent_types[-3:]
            idx_mn = recent_indices[-3:]
            
            # Minor bullish ChoCh
            if types_mn[0] == 'H' and types_mn[1] == 'L' and types_mn[2] == 'H':
                if high[idx_mn[2]] > high[idx_mn[0]] and low[idx_mn[1]] < low[idx_mn[0]]:
                    minor_bull_chosh[idx_mn[2]] = True
            
            # Minor bearish ChoCh
            if types_mn[0] == 'L' and types_mn[1] == 'H' and types_mn[2] == 'L':
                if low[idx_mn[2]] < low[idx_mn[0]] and high[idx_mn[1]] > high[idx_mn[0]]:
                    minor_bear_chosh[idx_mn[2]] = True
            
            # Minor Bullish BoS
            if types_mn[1] == 'L' and types_mn[2] == 'H':
                if high[idx_mn[2]] > high[idx_mn[0]]:
                    minor_bull_bos[idx_mn[2]] = True
            
            # Minor Bearish BoS
            if types_mn[1] == 'H' and types_mn[2] == 'L':
                if low[idx_mn[2]] < low[idx_mn[0]]:
                    minor_bear_bos[idx_mn[2]] = True
    
    major_bull_chosh = pd.Series(major_bull_chosh, index=df.index)
    major_bear_chosh = pd.Series(major_bear_chosh, index=df.index)
    major_bull_bos = pd.Series(major_bull_bos, index=df.index)
    major_bear_bos = pd.Series(major_bear_bos, index=df.index)
    minor_bull_chosh = pd.Series(minor_bull_chosh, index=df.index)
    minor_bear_chosh = pd.Series(minor_bear_chosh, index=df.index)
    minor_bull_bos = pd.Series(minor_bull_bos, index=df.index)
    minor_bear_bos = pd.Series(minor_bear_bos, index=df.index)
    
    # Trend detection
    trend_bull = pd.Series(False, index=df.index)
    trend_bear = pd.Series(False, index=df.index)
    
    for i in range(PP + 1, bar_count):
        if i < 3:
            continue
        
        recent_t = zz_type[-3:] if len(zz_type) >= 3 else zz_type
        recent_i = zz_index[-3:] if len(zz_index) >= 3 else zz_index
        
        if len(recent_t) >= 3:
            if recent_t[-1] == 'H' and recent_t[-2] == 'L':
                if len(recent_t) >= 3 and high[recent_i[-1]] > high[recent_i[-3]]:
                    trend_bull.iloc[i] = True
            
            if recent_t[-1] == 'L' and recent_t[-2] == 'H':
                if len(recent_t) >= 3 and low[recent_i[-1]] < low[recent_i[-3]]:
                    trend_bear.iloc[i] = True
    
    # Entry conditions
    bull_msignal = pd.Series(False, index=df.index)
    bear_msignal = pd.Series(False, index=df.index)
    
    for i in range(PP + 1, bar_count):
        bull_condition = (major_bull_bos.iloc[i] or major_bull_chosh.iloc[i] or
                         minor_bull_bos.iloc[i] or minor_bull_chosh.iloc[i])
        bear_condition = (major_bear_bos.iloc[i] or major_bear_chosh.iloc[i] or
                         minor_bear_bos.iloc[i] or minor_bear_chosh.iloc[i])
        
        if bull_condition:
            bull_msignal.iloc[i] = True
        if bear_condition:
            bear_msignal.iloc[i] = True
    
    # Entry signal conditions with pullback
    dt_long_entry = pd.Series(False, index=df.index)
    db_short_entry = pd.Series(False, index=df.index)
    
    for i in range(PP + 10, bar_count):
        if bull_msignal.iloc[i]:
            recent_highs = []
            for j in range(i - 1, max(PP, i - 20), -1):
                if pivot_high.iloc[j]:
                    recent_highs.append((high.iloc[j], j))
                    break
            
            for j in range(i - 1, max(PP, i - 20), -1):
                if pivot_low.iloc[j] and low.iloc[j] < high.iloc[i]:
                    dt_long_entry.iloc[i] = True
                    break
                    if len(recent_highs) > 0 and low.iloc[j] < recent_highs[0][0]:
                        dt_long_entry.iloc[i] = True
                        break
        
        if bear_msignal.iloc[i]:
            for j in range(i - 1, max(PP, i - 20), -1):
                if pivot_high.iloc[j] and high.iloc[j] > low.iloc[i]:
                    db_short_entry.iloc[i] = True
                    break
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(PP + 10, bar_count):
        if pd.isna(high.iloc[i]) or pd.isna(low.iloc[i]) or pd.isna(close.iloc[i]):
            continue
        
        direction = None
        if dt_long_entry.iloc[i]:
            direction = 'long'
        elif db_short_entry.iloc[i]:
            direction = 'short'
        
        if direction is not None:
            ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries