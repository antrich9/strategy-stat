import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['datetime'].dt.hour
    
    is7am = (df['hour'] >= 7) & (df['hour'] < 11)
    is11am = (df['hour'] >= 11) & (df['hour'] < 15)
    is3pm = (df['hour'] >= 15) & (df['hour'] < 19)
    
    crtHigh = pd.Series(np.nan, index=df.index)
    crtLow = pd.Series(np.nan, index=df.index)
    
    mask_7am = is7am
    crtHigh[mask_7am] = df.loc[mask_7am, 'high']
    crtLow[mask_7am] = df.loc[mask_7am, 'low']
    
    crtHigh = crtHigh.ffill()
    crtLow = crtLow.ffill()
    
    breakHigh = df['high'] > crtHigh
    breakLow = df['low'] < crtLow
    
    fvgCondition = (df['low'].shift(1) > df['high'].shift(2)) & (df['high'] < df['low'].shift(1))
    
    longCondition = breakLow & fvgCondition
    shortCondition = breakHigh & fvgCondition
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(crtHigh.iloc[i]) or pd.isna(crtLow.iloc[i]):
            continue
            
        if longCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        
        if shortCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries