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
    PP = 5
    atrLength = 55
    
    ATR = df['high'].rolling(atrLength).max() - df['low'].rolling(atrLength).min()
    ATR = ATR.replace(0, np.nan).ffill()
    ATR = ATR.rolling(atrLength).mean()
    
    pivothigh = pd.Series(index=df.index, dtype=float)
    pivotlow = pd.Series(index=df.index, dtype=float)
    
    for i in range(PP, len(df) - PP):
        if df['high'].iloc[i] == df['high'].iloc[i-PP:i+PP+1].max():
            pivothigh.iloc[i] = df['high'].iloc[i]
        if df['low'].iloc[i] == df['low'].iloc[i-PP:i+PP+1].min():
            pivotlow.iloc[i] = df['low'].iloc[i]
    
    pivothigh = pivothigh.fillna(0)
    pivotlow = pivotlow.fillna(0)
    
    zigzag_type = []
    zigzag_value = []
    zigzag_index = []
    
    for i in range(PP, len(df)):
        if pivothigh.iloc[i] > 0 or pivotlow.iloc[i] > 0:
            if len(zigzag_type) == 0:
                if pivothigh.iloc[i] > 0:
                    zigzag_type.append('H')
                    zigzag_value.append(df['high'].iloc[i])
                    zigzag_index.append(i)
                else:
                    zigzag_type.append('L')
                    zigzag_value.append(df['low'].iloc[i])
                    zigzag_index.append(i)
            else:
                last_type = zigzag_type[-1]
                if last_type in ['L', 'LL']:
                    if pivotlow.iloc[i] > 0 and df['low'].iloc[i] < zigzag_value[-1]:
                        zigzag_type.pop()
                        zigzag_value.pop()
                        zigzag_index.pop()
                        zigzag_type.append('LL')
                        zigzag_value.append(df['low'].iloc[i])
                        zigzag_index.append(i)
                    elif pivothigh.iloc[i] > 0:
                        zigzag_type.append('H')
                        zigzag_value.append(df['high'].iloc[i])
                        zigzag_index.append(i)
                elif last_type in ['H', 'HH']:
                    if pivothigh.iloc[i] > 0 and df['high'].iloc[i] > zigzag_value[-1]:
                        zigzag_type.pop()
                        zigzag_value.pop()
                        zigzag_index.pop()
                        zigzag_type.append('HH')
                        zigzag_value.append(df['high'].iloc[i])
                        zigzag_index.append(i)
                    elif pivotlow.iloc[i] > 0:
                        zigzag_type.append('L')
                        zigzag_value.append(df['low'].iloc[i])
                        zigzag_index.append(i)
    
    boS_bull = pd.Series(False, index=df.index)
    boS_bear = pd.Series(False, index=df.index)
    choCh_bull = pd.Series(False, index=df.index)
    choCh_bear = pd.Series(False, index=df.index)
    dtTradeTriggered = pd.Series(False, index=df.index)
    dbTradeTriggered = pd.Series(False, index=df.index)
    
    for i in range(PP + 1, len(df)):
        for j in range(len(zigzag_type) - 1, 0, -1):
            if len(zigzag_type) > j and zigzag_index[j] <= i:
                curr_type = zigzag_type[j]
                prev_type = zigzag_type[j - 1]
                curr_val = zigzag_value[j]
                prev_val = zigzag_value[j - 1]
                
                if curr_type in ['H', 'HH'] and prev_type in ['L', 'LL']:
                    if curr_val > prev_val:
                        boS_bull.iloc[i] = True
                elif curr_type in ['L', 'LL'] and prev_type in ['H', 'HH']:
                    if curr_val < prev_val:
                        boS_bear.iloc[i] = True
                break
        
        for j in range(len(zigzag_type) - 1, 2, -1):
            if len(zigzag_type) > j and zigzag_index[j] <= i:
                curr_type = zigzag_type[j]
                prev_type = zigzag_type[j - 1]
                prev2_type = zigzag_type[j - 2]
                
                if curr_type in ['H', 'HH'] and prev_type in ['L', 'LL'] and prev2_type in ['H', 'HH']:
                    boS_bull.iloc[i] = True
                    choCh_bear.iloc[i] = True
                elif curr_type in ['L', 'LL'] and prev_type in ['H', 'HH'] and prev2_type in ['L', 'LL']:
                    boS_bear.iloc[i] = True
                    choCh_bull.iloc[i] = True
                break
    
    for i in range(PP + 1, len(df)):
        if boS_bull.iloc[i] or choCh_bull.iloc[i]:
            dtTradeTriggered.iloc[i] = True
        if boS_bear.iloc[i] or choCh_bear.iloc[i]:
            dbTradeTriggered.iloc[i] = True
    
    entries = []
    trade_num = 1
    
    for i in range(PP, len(df)):
        if pd.isna(ATR.iloc[i]):
            continue
        
        direction = None
        if dtTradeTriggered.iloc[i]:
            direction = 'long'
        elif dbTradeTriggered.iloc[i]:
            direction = 'short'
        
        if direction is None:
            continue
        
        entry_ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        entry_price = float(df['close'].iloc[i])
        
        entries.append({
            'trade_num': trade_num,
            'direction': direction,
            'entry_ts': entry_ts,
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