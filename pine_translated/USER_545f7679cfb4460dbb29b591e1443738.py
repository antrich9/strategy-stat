import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('datetime', inplace=True)
    
    data_4h = df.resample('240T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    
    data_4h['time'] = data_4h['time'].astype(np.int64)
    
    vol_sma = data_4h['volume'].rolling(9).mean()
    volfilt = data_4h['volume'] > vol_sma * 1.5
    
    tr1 = data_4h['high'] - data_4h['low']
    tr2 = (data_4h['high'] - data_4h['close'].shift()).abs()
    tr3 = (data_4h['low'] - data_4h['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    
    atrfilt = ((data_4h['low'] - data_4h['high'].shift(2)) > atr) | ((data_4h['low'].shift(2) - data_4h['high']) > atr)
    
    loc = data_4h['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    bfvg = (data_4h['low'] > data_4h['high'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (data_4h['high'] < data_4h['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    lastFVG = 0
    trade_num = 1
    entries = []
    
    for i in range(len(data_4h)):
        if bfvg.iloc[i] and lastFVG == -1:
            ts = int(data_4h['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(data_4h['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(data_4h['close'].iloc[i]),
                'raw_price_b': float(data_4h['close'].iloc[i])
            })
            trade_num += 1
        elif sfvg.iloc[i] and lastFVG == 1:
            ts = int(data_4h['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(data_4h['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(data_4h['close'].iloc[i]),
                'raw_price_b': float(data_4h['close'].iloc[i])
            })
            trade_num += 1
        
        if bfvg.iloc[i]:
            lastFVG = 1
        elif sfvg.iloc[i]:
            lastFVG = -1
    
    return entries