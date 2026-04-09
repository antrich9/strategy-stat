import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Resample to 4H
    df_4h = df.copy()
    df_4h['time'] = pd.to_datetime(df_4h['time'], unit='s', utc=True)
    df_4h.set_index('time', inplace=True)
    
    # Resample to 4H (240 minutes)
    ohlc_4h = df_4h.resample('240T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    ohlc_4h.dropna(inplace=True)
    ohlc_4h.reset_index(inplace=True)
    
    # Calculate indicators
    # SMA 54 for trend
    ohlc_4h['sma54'] = ohlc_4h['close'].rolling(54).mean()
    
    # ATR 20 for filter
    high_low = ohlc_4h['high'] - ohlc_4h['low']
    high_close = np.abs(ohlc_4h['high'] - ohlc_4h['close'].shift())
    low_close = np.abs(ohlc_4h['low'] - ohlc_4h['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    ohlc_4h['atr20'] = tr.rolling(20).mean()
    
    # Volume SMA 9
    ohlc_4h['vol_sma9'] = ohlc_4h['volume'].rolling(9).mean()
    
    # Filters
    ohlc_4h['volfilt'] = ohlc_4h['volume'].shift(1) > ohlc_4h['vol_sma9'] * 1.5
    ohlc_4h['atr20_adj'] = ohlc_4h['atr20'] / 1.5
    ohlc_4h['atrfilt'] = ((ohlc_4h['low'] - ohlc_4h['high'].shift(2) > ohlc_4h['atr20_adj']) | 
                          (ohlc_4h['low'].shift(2) - ohlc_4h['high'] > ohlc_4h['atr20_adj']))
    
    # Trend filters
    ohlc_4h['loc2'] = ohlc_4h['sma54'] > ohlc_4h['sma54'].shift(1)
    ohlc_4h['locfiltb'] = ohlc_4h['loc2']
    ohlc_4h['locfilts'] = ~ohlc_4h['loc2']
    
    # FVGs
    ohlc_4h['bfvg'] = (ohlc_4h['low'] > ohlc_4h['high'].shift(2)) & ohlc_4h['volfilt'] & ohlc_4h['atrfilt'] & ohlc_4h['locfiltb']
    ohlc_4h['sfvg'] = (ohlc_4h['high'] < ohlc_4h['low'].shift(2)) & ohlc_4h['volfilt'] & ohlc_4h['atrfilt'] & ohlc_4h['locfilts']
    
    # Sharp turn detection
    entries = []
    trade_num = 1
    lastFVG = 0  # 0: none, 1: bullish, -1: bearish
    
    for i in range(len(ohlc_4h)):
        # Skip if indicators are NaN
        if pd.isna(ohlc_4h['sma54'].iloc[i]) or pd.isna(ohlc_4h['atr20'].iloc[i]):
            continue
            
        bfvg = ohlc_4h['bfvg'].iloc[i]
        sfvg = ohlc_4h['sfvg'].iloc[i]
        
        if bfvg and lastFVG == -1:
            # Bullish sharp turn - LONG entry
            ts = int(ohlc_4h['time'].iloc[i].timestamp())
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': ohlc_4h['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': ohlc_4h['close'].iloc[i],
                'raw_price_b': ohlc_4h['close'].iloc[i]
            })
            trade_num += 1
            lastFVG = 1
        elif sfvg and lastFVG == 1:
            # Bearish sharp turn - SHORT entry
            ts = int(ohlc_4h['time'].iloc[i].timestamp())
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': ohlc_4h['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': ohlc_4h['close'].iloc[i],
                'raw_price_b': ohlc_4h['close'].iloc[i]
            })
            trade_num += 1
            lastFVG = -1
        elif bfvg:
            lastFVG = 1
        elif sfvg:
            lastFVG = -1
            
    return entries