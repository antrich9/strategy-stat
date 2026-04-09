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
    
    # Parameters (all inputs were fixed in Pine Script)
    length = 200
    HV_ma = 20
    divisor = 3.6
    
    # Calculate range
    range_1 = df['high'] - df['low']
    
    # Simple Moving Averages
    rangeAvg = range_1.rolling(length).mean()
    durchschnitt = df['volume'].rolling(HV_ma).mean()
    volumeA = df['volume'].rolling(length).mean()
    
    # Previous bar values
    high1 = df['high'].shift(1)
    low1 = df['low'].shift(1)
    mid1 = ((df['high'] + df['low']) / 2).shift(1)
    
    # Calculate u1 and d1
    u1 = mid1 + (high1 - low1) / divisor
    d1 = mid1 - (high1 - low1) / divisor
    
    # Red (short) conditions
    r_enabled1 = (range_1 > rangeAvg) & (df['close'] < d1) & (df['volume'] > volumeA)
    r_enabled2 = df['close'] < mid1
    r_enabled = r_enabled1 | r_enabled2
    
    # Green (long) conditions
    g_enabled1 = df['close'] > mid1
    g_enabled2 = (range_1 > rangeAvg) & (df['close'] > u1) & (df['volume'] > volumeA)
    g_enabled3 = (df['high'] > high1) & (range_1 < rangeAvg / 1.5) & (df['volume'] < volumeA)
    g_enabled4 = (df['low'] < low1) & (range_1 < rangeAvg / 1.5) & (df['volume'] > volumeA)
    g_enabled = g_enabled1 | g_enabled2 | g_enabled3 | g_enabled4
    
    # Basic conditions with volume filter
    basicLongHEVCondition = g_enabled & (df['volume'] > durchschnitt)
    basicShorHEVondition = r_enabled & (df['volume'] > durchschnitt)
    
    # All inputs are fixed: useHEV=true, crossHEV=true, highlightMovementsHEV=true, inverseHEV=false
    HEVSignalsLongCross = (~basicLongHEVCondition.shift(1).fillna(False)) & basicLongHEVCondition
    HEVSignalsShorHEVross = (~basicShorHEVondition.shift(1).fillna(False)) & basicShorHEVondition
    
    HEVSignalsLongFinal = HEVSignalsLongCross
    HEVSignalsShortFinal = HEVSignalsShorHEVross
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(rangeAvg.iloc[i]) or pd.isna(durchschnitt.iloc[i]) or pd.isna(volumeA.iloc[i]):
            continue
        if pd.isna(HEVSignalsLongFinal.iloc[i]) or pd.isna(HEVSignalsShortFinal.iloc[i]):
            continue
        
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = float(df['close'].iloc[i])
        
        if HEVSignalsLongFinal.iloc[i]:
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
        
        if HEVSignalsShortFinal.iloc[i]:
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