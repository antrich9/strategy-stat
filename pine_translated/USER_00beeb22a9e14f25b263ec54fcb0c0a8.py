import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.

    Returns list of dicts:
    [{'trade_num': int, 'direction': 'long' or 'short',
      'entry_ts': int, 'entry_time': str,
      'entry_price_guess': float,
      'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
      'raw_price_a': float, 'raw_price_b': float}]
    """
    
    threshold = 0.0 / 100
    consecutiveBullCount = 0
    consecutiveBearCount = 0
    prevBullFvg = False
    prevBearFvg = False
    
    entries = []
    trade_num = 1
    
    atr = np.zeros(len(df))
    tr = np.zeros(len(df))
    tr[0] = df['high'].iloc[0] - df['low'].iloc[0]
    atr[0] = df['high'].iloc[0] - df['low'].iloc[0]
    for i in range(1, len(df)):
        tr[i] = max(df['high'].iloc[i] - df['low'].iloc[i], 
                    max(abs(df['high'].iloc[i] - df['close'].iloc[i-1]), 
                        abs(df['low'].iloc[i] - df['close'].iloc[i-1])))
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    df['atr'] = atr
    
    df['date'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.date
    daily_ohlc = df.groupby('date').agg({'high': 'max', 'low': 'min'}).reset_index()
    daily_ohlc['date'] = pd.to_datetime(daily_ohlc['date'])
    daily_ohlc['prev_day_high'] = daily_ohlc['high'].shift(1)
    daily_ohlc['prev_day_low'] = daily_ohlc['low'].shift(1)
    
    df = df.merge(daily_ohlc[['date', 'prev_day_high', 'prev_day_low']], on='date', how='left')
    
    for i in range(3, len(df)):
        if i < 2:
            continue
        if pd.isna(df['high'].iloc[i-2]) or pd.isna(df['low'].iloc[i-2]):
            continue
        if pd.isna(df['atr'].iloc[i]):
            continue
        
        bull_fvg = df['low'].iloc[i] > df['high'].iloc[i-2] and df['close'].iloc[i-1] > df['high'].iloc[i-2] and (df['low'].iloc[i] - df['high'].iloc[i-2]) / df['high'].iloc[i-2] > threshold
        bear_fvg = df['high'].iloc[i] < df['low'].iloc[i-2] and df['close'].iloc[i-1] < df['low'].iloc[i-2] and (df['low'].iloc[i-2] - df['high'].iloc[i]) / df['high'].iloc[i] > threshold
        
        if bull_fvg and not prevBullFvg:
            consecutiveBullCount += 1
            consecutiveBearCount = 0
        elif not bull_fvg and prevBullFvg:
            consecutiveBearCount = 0
        
        if bear_fvg and not prevBearFvg:
            consecutiveBearCount += 1
            consecutiveBullCount = 0
        elif not bear_fvg and prevBearFvg:
            consecutiveBullCount = 0
        
        if consecutiveBullCount >= 2:
            ts = int(df['time'].iloc[i])
            entry_price_guess = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1
            consecutiveBullCount = 0
        
        if consecutiveBearCount >= 2:
            ts = int(df['time'].iloc[i])
            entry_price_guess = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1
            consecutiveBearCount = 0
        
        prevBullFvg = bull_fvg
        prevBearFvg = bear_fvg
    
    return entries