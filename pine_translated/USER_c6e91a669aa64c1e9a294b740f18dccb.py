import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.sort_values('time').reset_index(drop=True)
    
    # Resample to 4H
    def resample_to_4h(data):
        data = data.copy()
        data.set_index('time', inplace=True)
        resampled = data.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        return resampled.reset_index()
    
    df_4h = resample_to_4h(df)
    df_4h.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    
    # ATR calculation
    high = df_4h['high']
    low = df_4h['low']
    close = df_4h['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder ATR
    atr_length = 20
    atr = tr.ewm(alpha=1/atr_length, adjust=False).mean()
    
    # Trend filter (SMA 54 on 4H)
    sma_54 = close.rolling(54).mean()
    trend_up = sma_54 > sma_54.shift(1)
    
    # FVG detection on 4H
    bullish_fvg = (low > high.shift(2)) & (tr > atr * 1.5) & trend_up
    bearish_fvg = (high < low.shift(2)) & (tr > atr * 1.5) & ~trend_up
    
    # Sharp turn detection
    bullish_sharp_turn = bullish_fvg & bearish_fvg.shift(1)
    bearish_sharp_turn = bearish_fvg & bullish_fvg.shift(1)
    
    # Generate signals
    signals = []
    for i in range(len(df_4h)):
        if bullish_sharp_turn.iloc[i]:
            signals.append({
                'timestamp': df_4h.iloc[i]['time'],
                'direction': 'long',
                'price': df_4h.iloc[i]['close']
            })
        elif bearish_sharp_turn.iloc[i]:
            signals.append({
                'timestamp': df_4h.iloc[i]['time'],
                'direction': 'short',
                'price': df_4h.iloc[i]['close']
            })
    
    return signals