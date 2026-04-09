import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('datetime', inplace=True)
    
    ohlc_4h = df[['open', 'high', 'low', 'close', 'volume']].resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    high_4h = ohlc_4h['high']
    low_4h = ohlc_4h['low']
    close_4h = ohlc_4h['close']
    volume_4h = ohlc_4h['volume']
    
    volfilt = volume_4h.shift(1) > volume_4h.rolling(9).mean() * 1.5
    volfilt = volfilt.fillna(True)
    
    high_low = high_4h - low_4h
    high_close = np.abs(high_4h - close_4h.shift(1))
    low_close = np.abs(low_4h - close_4h.shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_4h = tr.ewm(alpha=1/14, adjust=False).mean() / 1.5
    
    atrfilt = ((low_4h - high_4h.shift(2) > atr_4h) | (low_4h.shift(2) - high_4h > atr_4h))
    atrfilt = atrfilt.fillna(True)
    
    loc1 = close_4h.rolling(54).mean()
    loc2 = loc1 > loc1.shift(1)
    locfiltb = loc2.fillna(True)
    locfilts = ~loc2
    locfilts = locfilts.fillna(True)
    
    bfvg = (low_4h > high_4h.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (high_4h < low_4h.shift(2)) & volfilt & atrfilt & locfilts
    
    lastFVG = np.nan
    entries = []
    trade_num = 1
    
    for i in range(len(ohlc_4h)):
        if np.isnan(lastFVG):
            if bfvg.iloc[i]:
                lastFVG = 1
            elif sfvg.iloc[i]:
                lastFVG = -1
        elif lastFVG == -1 and bfvg.iloc[i]:
            ts = int(ohlc_4h.index[i].timestamp())
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': close_4h.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close_4h.iloc[i],
                'raw_price_b': close_4h.iloc[i]
            })
            trade_num += 1
            lastFVG = 1
        elif lastFVG == 1 and sfvg.iloc[i]:
            ts = int(ohlc_4h.index[i].timestamp())
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': close_4h.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close_4h.iloc[i],
                'raw_price_b': close_4h.iloc[i]
            })
            trade_num += 1
            lastFVG = -1
        elif bfvg.iloc[i]:
            lastFVG = 1
        elif sfvg.iloc[i]:
            lastFVG = -1
    
    return entries