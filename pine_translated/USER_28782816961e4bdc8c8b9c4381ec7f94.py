import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Resample to 4H
    ohlc_4h = df.resample('240T').agg({
        'time': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Calculate volume filter on 4H
    vol_sma_4h = ohlc_4h['volume'].rolling(9).mean()
    
    # Calculate ATR filter on 4H (Wilder method)
    tr1 = ohlc_4h['high'] - ohlc_4h['low']
    tr2 = abs(ohlc_4h['high'] - ohlc_4h['close'].shift(1))
    tr3 = abs(ohlc_4h['low'] - ohlc_4h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = np.zeros(len(tr))
    atr_4h[19] = tr.iloc[:20].mean()
    for i in range(20, len(tr)):
        atr_4h[i] = (atr_4h[i-1] * 19 + tr.iloc[i]) / 20
    atr_4h = pd.Series(atr_4h, index=tr.index) / 1.5
    
    # Calculate trend filter on 4H
    loc = ohlc_4h['close'].rolling(54).mean()
    loc_prev = loc.shift(1)
    
    # Calculate FVG conditions on 4H
    volfilt1 = vol_sma_4h * 1.5
    atrfilt_cond = (ohlc_4h['low'] - ohlc_4h['high'].shift(2) > atr_4h) | (ohlc_4h['low'].shift(2) - ohlc_4h['high'] > atr_4h)
    
    locfiltb1 = loc > loc_prev
    locfilts1 = loc < loc_prev
    
    bfvg1 = (ohlc_4h['low'] > ohlc_4h['high'].shift(2)) & (ohlc_4h['volume'] > volfilt1) & atrfilt_cond & locfiltb1
    sfvg1 = (ohlc_4h['high'] < ohlc_4h['low'].shift(2)) & (ohlc_4h['volume'] > volfilt1) & atrfilt_cond & locfilts1
    
    # Detect new 4H candles
    is_new_4h1 = ohlc_4h.index.to_series().diff().dt.total_seconds() > 240 * 60
    
    # Track last FVG state
    lastFVG = 0
    trade_num = 1
    entries = []
    
    for i in range(1, len(ohlc_4h)):
        if is_new_4h1.iloc[i] and i >= 20:
            bullish_fvg = bfvg1.iloc[i]
            bearish_fvg = sfvg1.iloc[i]
            
            if bullish_fvg and lastFVG == -1:
                entry_ts = int(ohlc_4h['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entry_price_guess = float(ohlc_4h['close'].iloc[i])
                
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': entry_price_guess,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price_guess,
                    'raw_price_b': entry_price_guess
                })
                trade_num += 1
                lastFVG = 1
            elif bearish_fvg and lastFVG == 1:
                entry_ts = int(ohlc_4h['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entry_price_guess = float(ohlc_4h['close'].iloc[i])
                
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': entry_price_guess,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price_guess,
                    'raw_price_b': entry_price_guess
                })
                trade_num += 1
                lastFVG = -1
            elif bullish_fvg:
                lastFVG = 1
            elif bearish_fvg:
                lastFVG = -1
    
    return entries