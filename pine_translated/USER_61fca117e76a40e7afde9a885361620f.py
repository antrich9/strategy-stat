import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df['day'] = df['datetime'].dt.date
    
    daily_hl = df.groupby('day').agg(
        pdHigh=('high', 'max'),
        pdLow=('low', 'min')
    ).shift(1)
    
    df = df.merge(daily_hl, left_on='day', right_index=True, how='left')
    
    df['newDay'] = df['day'] != df['day'].shift(1)
    
    df['isUp'] = df['close'] > df['open']
    df['isDown'] = df['close'] < df['open']
    
    df['bullishOB'] = df['isDown'].shift(1) & df['isUp'] & (df['close'] > df['high'].shift(1))
    df['bearishOB'] = df['isUp'].shift(1) & df['isDown'] & (df['close'] < df['low'].shift(1))
    df['bullishFVG'] = df['low'] > df['high'].shift(2)
    df['bearishFVG'] = df['high'] < df['low'].shift(2)
    
    df['bullishSignal'] = df['bullishOB'] & df['bullishFVG']
    df['bearishSignal'] = df['bearishOB'] & df['bearishFVG']
    
    entries = []
    trade_num = 1
    sweptHigh = False
    sweptLow = False
    
    for i in range(3, len(df)):
        if df.loc[df.index[i], 'newDay']:
            sweptHigh = False
            sweptLow = False
        
        if not sweptHigh and df['high'].iloc[i] > df['pdHigh'].iloc[i]:
            sweptHigh = True
            if df['bullishSignal'].iloc[i]:
                ts = int(df['time'].iloc[i])
                entry_price = float(df['close'].iloc[i])
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
                trade_num += 1
        
        if not sweptLow and df['low'].iloc[i] < df['pdLow'].iloc[i]:
            sweptLow = True
            if df['bearishSignal'].iloc[i]:
                ts = int(df['time'].iloc[i])
                entry_price = float(df['close'].iloc[i])
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
                trade_num += 1
    
    return entries