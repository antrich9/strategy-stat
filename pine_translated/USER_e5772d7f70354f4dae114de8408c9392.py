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
    
    # Default input values from Pine Script
    useTC = True
    crossTC = True
    inverseTC = False
    highlightMovementsTC = True
    srcTC = df['close'].copy()
    
    lengthTC1 = 5
    factorTC1 = 0.7
    lengthTC2 = 18
    factorTC2 = 0.7
    
    # Triple Crown calculation function
    def gdTC(src, length, factor):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factor) - ema2 * factor
    
    # Calculate TC1 and TC2 with triple nesting
    TC1 = gdTC(gdTC(gdTC(srcTC, lengthTC1, factorTC1), lengthTC1, factorTC1), lengthTC1, factorTC1)
    TC2 = gdTC(gdTC(gdTC(srcTC, lengthTC2, factorTC2), lengthTC2, factorTC2), lengthTC2, factorTC2)
    
    # Shift for previous bar values
    TC1_prev = TC1.shift(1)
    TC2_prev = TC2.shift(1)
    
    # Basic conditions
    basicLongCondition = (TC1 > TC1_prev) & (TC2 > TC2_prev) & (TC1 > TC2)
    basicShortCondition = (TC1 < TC1_prev) & (TC2 < TC2_prev) & (TC1 < TC2)
    
    # TCSignals based on useTC and highlightMovementsTC
    if useTC:
        if highlightMovementsTC:
            TCSignalsLong = basicLongCondition
            TCSignalsShort = basicShortCondition
        else:
            TCSignalsLong = df['close'] > TC1
            TCSignalsShort = df['close'] < TC1
    else:
        TCSignalsLong = pd.Series(True, index=df.index)
        TCSignalsShort = pd.Series(True, index=df.index)
    
    # Shift for previous bar
    TCSignalsLong_prev = TCSignalsLong.shift(1)
    TCSignalsShort_prev = TCSignalsShort.shift(1)
    
    # Cross confirmation
    if crossTC:
        TCSignalsLongCross = (~TCSignalsLong_prev) & TCSignalsLong
        TCSignalsShortCross = (~TCSignalsShort_prev) & TCSignalsShort
    else:
        TCSignalsLongCross = TCSignalsLong
        TCSignalsShortCross = TCSignalsShort
    
    # Final signals based on inverseTC
    if inverseTC:
        TCSignalsLongFinal = TCSignalsShortCross
        TCSignalsShortFinal = TCSignalsLongCross
    else:
        TCSignalsLongFinal = TCSignalsLongCross
        TCSignalsShortFinal = TCSignalsShortCross
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        # Skip if indicators are NaN
        if pd.isna(TC1.iloc[i]) or pd.isna(TC2.iloc[i]):
            continue
        if pd.isna(TC1_prev.iloc[i]) or pd.isna(TC2_prev.iloc[i]):
            continue
        if i > 0 and (pd.isna(TCSignalsLong.iloc[i-1]) or pd.isna(TCSignalsShort.iloc[i-1])):
            continue
        
        entry_price = df['close'].iloc[i]
        ts = df['time'].iloc[i]
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        # Long entry
        if TCSignalsLongFinal.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
        
        # Short entry
        if TCSignalsShortFinal.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
    
    return entries