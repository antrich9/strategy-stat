import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    high = df['high']
    low = df['low']
    close = df['close']
    n = len(df)
    entries = []
    trade_num = 0
    
    ig_active = False
    ig_c1_high = np.nan
    ig_c1_low = np.nan
    ig_c3_high = np.nan
    ig_c3_low = np.nan
    ig_direction = 0
    ig_validation_end = 0
    
    for i in range(2, n):
        bullishFVG1 = high.iloc[i-2] < low.iloc[i]
        bearishFVG1 = low.iloc[i-2] > high.iloc[i]
        
        if not ig_active:
            if bullishFVG1:
                ig_active = True
                ig_direction = 1
                ig_c1_high = high.iloc[i-2]
                ig_c1_low = low.iloc[i-2]
                ig_c3_high = high.iloc[i]
                ig_c3_low = low.iloc[i]
                ig_validation_end = i + 4
            elif bearishFVG1:
                ig_active = True
                ig_direction = -1
                ig_c1_high = high.iloc[i-2]
                ig_c1_low = low.iloc[i-2]
                ig_c3_high = high.iloc[i]
                ig_c3_low = low.iloc[i]
                ig_validation_end = i + 4
        
        validated = False
        if ig_active and i <= ig_validation_end:
            if ig_direction == 1 and close.iloc[i] < ig_c1_high:
                validated = True
            if ig_direction == -1 and close.iloc[i] > ig_c1_low:
                validated = True
        
        if validated:
            trade_num += 1
            entry_price = close.iloc[i]
            ts = int(timestamps.iloc[i])
            if ig_direction == -1:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
            else:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
            ig_active = False
        
        if ig_active and i > ig_validation_end:
            ig_active = False
    
    return entries