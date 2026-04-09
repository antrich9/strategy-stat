import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.
    """
    entries = []
    trade_num = 1
    
    # Resample to 4H
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df_4h = df.set_index('time_dt').resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Identify new 4H candle starts on 15min chart
    df['hour'] = df['time_dt'].dt.hour
    df['minute'] = df['time_dt'].dt.minute
    is_new_4h = (df['hour'] % 4 == 0) & (df['minute'] == 0)
    
    # Calculate 4H indicators
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    close_4h = df_4h['close']
    volume_4h = df_4h['volume']
    
    # Volume filter
    vol_ma = volume_4h.rolling(9).mean() * 1.5
    vol_filt = volume_4h.shift(1) > vol_ma
    
    # ATR filter (length 20)
    tr_4h = pd.concat([high_4h, close_4h.shift(1)], axis=1).max(axis=1) - \
            pd.concat([low_4h, close_4h.shift(1)], axis=1).min(axis=1)
    atr_4h = tr_4h.rolling(20).mean() / 1.5
    
    # Trend filter
    sma_54 = close_4h.rolling(54).mean()
    trend_up = sma_54 > sma_54.shift(1)
    
    # Bullish FVG: low > high[2] (two 4H candles ago)
    bfvg = (low_4h > high_4h.shift(2)) & vol_filt & atr_4h & trend_up
    
    # Bearish FVG: high < low[2]
    sfvg = (high_4h < low_4h.shift(2)) & vol_filt & atr_4h & ~trend_up
    
    # Sharp Turn detection
    last_fvg = 0
    for i in range(len(df_4h)):
        if is_new_4h.iloc[i]:
            if bfvg.iloc[i] and last_fvg == -1:
                # Bullish Sharp Turn
                entry = {
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': df['time'].iloc[i],
                    'entry_price': df['close'].iloc[i],
                    'raw_price_a': low_4h.iloc[i],
                    'raw_price_b': high_4h.iloc[i-2]
                }
                entries.append(entry)
                trade_num += 1
            elif sfvg.iloc[i] and last_fvg == 1:
                # Bearish Sharp Turn
                entry = {
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': df['time'].iloc[i],
                    'entry_price': df['close'].iloc[i],
                    'raw_price_a': high_4h.iloc[i],
                    'raw_price_b': low_4h.iloc[i-2]
                }
                entries.append(entry)
                trade_num += 1
            
            if bfvg.iloc[i]:
                last_fvg = 1
            elif sfvg.iloc[i]:
                last_fvg = -1
    
    return entries