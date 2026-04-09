import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    fvg_isNew = False
    entered = False
    fvg_top = np.nan
    fvg_bottom = np.nan
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        ts = df['time'].iloc[i]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        
        # Session: 1500-1559 GMT every day
        in_session = dt.hour == 15
        
        # Reset FVG and entry flag after session ends
        if not in_session:
            fvg_isNew = False
            entered = False
        
        # Detect first FVG in NY 15:00-15:59 session
        if in_session and not fvg_isNew and i > 0:
            fvg_top = df['high'].iloc[i-1]
            fvg_bottom = df['low'].iloc[i-1]
            fvg_isNew = True
        
        # Detect bearish FVG (3 candle pattern)
        if i >= 2:
            bear_fvg = df['high'].iloc[i-2] < df['low'].iloc[i]
        else:
            bear_fvg = False
        
        # Entry on retrace to 1st FVG
        if bear_fvg and fvg_isNew and not entered:
            entry_price = (fvg_top + fvg_bottom) / 2
            atr_val = df['high'].iloc[i] - df['low'].iloc[i]
            stop_loss = entry_price + atr_val * 2
            take_profit = entry_price - atr_val * 3
            
            if stop_loss > entry_price and take_profit < entry_price:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
                entered = True
        
        # Reset entry flag when session ends
        if not in_session:
            entered = False
    
    return entries