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
    time = df['time']
    
    n = len(df)
    pivot_high = pd.Series(np.nan, index=df.index)
    pivot_low = pd.Series(np.nan, index=df.index)
    
    left_len = 5
    right_len = 5
    
    for i in range(left_len, n - right_len):
        window_high = high.iloc[i - left_len:i + right_len + 1]
        window_low = low.iloc[i - left_len:i + right_len + 1]
        if high.iloc[i] == window_high.max():
            pivot_high.iloc[i] = high.iloc[i]
        if low.iloc[i] == window_low.min():
            pivot_low.iloc[i] = low.iloc[i]
    
    Major_HighLevel = np.nan
    Major_LowLevel = np.nan
    Major_HighIndex = -1
    Major_LowIndex = -1
    LockBreak_M = -1
    ExternalTrend = 'No Trend'
    
    trade_num = 0
    entries = []
    
    for i in range(1, n):
        if not np.isnan(pivot_high.iloc[i]):
            Major_HighLevel = pivot_high.iloc[i]
            Major_HighIndex = i
        
        if not np.isnan(pivot_low.iloc[i]):
            Major_LowLevel = pivot_low.iloc[i]
            Major_LowIndex = i
        
        if i == 0:
            continue
            
        crossover_cond = close.iloc[i] > Major_HighLevel and close.iloc[i-1] <= Major_HighLevel
        crossunder_cond = close.iloc[i] < Major_LowLevel and close.iloc[i-1] >= Major_LowLevel
        
        if crossover_cond and LockBreak_M != Major_HighIndex:
            if ExternalTrend == 'No Trend' or ExternalTrend == 'Up Trend':
                trade_num += 1
                entry_price = close.iloc[i]
                entry_ts = int(time.iloc[i])
                entry_time_str = datetime.fromtimestamp(entry_ts / 1000.0, tz=timezone.utc).isoformat()
                
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time_str,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                
                LockBreak_M = Major_HighIndex
                ExternalTrend = 'Up Trend'
                
            elif ExternalTrend == 'Down Trend':
                LockBreak_M = Major_HighIndex
                ExternalTrend = 'Up Trend'
        
        if crossunder_cond and LockBreak_M != Major_LowIndex:
            if ExternalTrend == 'No Trend' or ExternalTrend == 'Down Trend':
                trade_num += 1
                entry_price = close.iloc[i]
                entry_ts = int(time.iloc[i])
                entry_time_str = datetime.fromtimestamp(entry_ts / 1000.0, tz=timezone.utc).isoformat()
                
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time_str,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                
                LockBreak_M = Major_LowIndex
                ExternalTrend = 'Down Trend'
                
            elif ExternalTrend == 'Up Trend':
                LockBreak_M = Major_LowIndex
                ExternalTrend = 'Down Trend'
    
    return entries