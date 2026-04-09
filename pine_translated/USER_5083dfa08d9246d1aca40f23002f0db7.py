import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    useXMA = True
    crossXMA = True
    inverseXMA = False
    periodXMA = 12
    porogXMA = 3.0
    
    close = df['close']
    
    mintick = max((df['high'] - df['low']).min(), 1e-10)
    
    ema = close.ewm(span=periodXMA, adjust=False).mean()
    emaCurrentXMA = ema
    emaPreviousXMA = ema.shift(1)
    
    signalXMA = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i == 0:
            continue
        if pd.notna(emaCurrentXMA.iloc[i]) and pd.notna(emaPreviousXMA.iloc[i]):
            if abs(emaCurrentXMA.iloc[i] - emaPreviousXMA.iloc[i]) >= porogXMA * mintick:
                signalXMA.iloc[i] = emaCurrentXMA.iloc[i]
    
    xmaSignalLong = close >= signalXMA
    xmaSignalShort = close <= signalXMA
    
    signalLongXMA = xmaSignalLong.copy()
    signalShortXMA = xmaSignalShort.copy()
    
    if useXMA and crossXMA:
        prevLong = xmaSignalLong.shift(1).fillna(False).astype(bool)
        currLong = xmaSignalLong.fillna(False).astype(bool)
        signalLongXMA = (prevLong == False) & (currLong == True)
        
        prevShort = xmaSignalShort.shift(1).fillna(False).astype(bool)
        currShort = xmaSignalShort.fillna(False).astype(bool)
        signalShortXMA = (prevShort == False) & (currShort == True)
    
    finalLongSignal = signalShortXMA if inverseXMA else signalLongXMA
    finalShortSignal = signalLongXMA if inverseXMA else signalShortXMA
    
    if not useXMA:
        finalLongSignal = pd.Series(True, index=df.index)
        finalShortSignal = pd.Series(True, index=df.index)
    
    results = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if pd.notna(finalLongSignal.iloc[i]) and finalLongSignal.iloc[i]:
            ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if pd.notna(finalShortSignal.iloc[i]) and finalShortSignal.iloc[i]:
            ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return results