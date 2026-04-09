import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    period = 20
    CandleType = True
    
    n = len(df)
    
    PH = pd.Series([np.nan] * n, index=df.index)
    PL = pd.Series([np.nan] * n, index=df.index)
    
    for i in range(period, n - period):
        left_start = i - period
        right_end = i + period + 1
        
        window_high = df['high'].iloc[left_start:right_end]
        window_low = df['low'].iloc[left_start:right_end]
        
        if df['high'].iloc[i] == window_high.max():
            PH.iloc[i] = df['high'].iloc[i]
        
        if df['low'].iloc[i] == window_low.min():
            PL.iloc[i] = df['low'].iloc[i]
    
    UpdatedHigh = float('-inf')
    UpdatedLow = float('inf')
    
    trade_num = 0
    entries = []
    
    ScrHigh = df['close'] if not CandleType else df['high']
    ScrLow = df['close'] if not CandleType else df['low']
    
    for i in range(1, n):
        prev_PH = PH.iloc[i-1] if not pd.isna(PH.iloc[i-1]) else UpdatedHigh
        curr_PH = PH.iloc[i] if not pd.isna(PH.iloc[i]) else prev_PH
        
        if curr_PH != prev_PH:
            UpdatedHigh = curr_PH
        
        prev_PL = PL.iloc[i-1] if not pd.isna(PL.iloc[i-1]) else UpdatedLow
        curr_PL = PL.iloc[i] if not pd.isna(PL.iloc[i]) else prev_PL
        
        if curr_PL != prev_PL:
            UpdatedLow = curr_PL
        
        if ScrLow.iloc[i] < UpdatedLow:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
        
        if ScrHigh.iloc[i] > UpdatedHigh:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
    
    return entries