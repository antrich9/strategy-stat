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
    
    rsiPeriod = 14
    rsiOverbought = 70.0
    rsiOversold = 30.0
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avgGain = gain.ewm(alpha=1.0/rsiPeriod, min_periods=rsiPeriod, adjust=False).mean()
    avgLoss = loss.ewm(alpha=1.0/rsiPeriod, min_periods=rsiPeriod, adjust=False).mean()
    
    rs = avgGain / avgLoss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    doubleTop = (
        (df['high'].shift(1) > df['high'].shift(0)) & 
        (df['high'].shift(1) > df['high'].shift(2)) &
        (rsi.shift(1) > rsiOverbought) &
        (rsi.shift(1) > rsi.shift(0)) &
        (rsi.shift(1) > rsi.shift(2))
    )
    
    doubleBottom = (
        (df['low'].shift(1) < df['low'].shift(0)) & 
        (df['low'].shift(1) < df['low'].shift(2)) &
        (rsi.shift(1) < rsiOversold) &
        (rsi.shift(1) < rsi.shift(0)) &
        (rsi.shift(1) < rsi.shift(2))
    )
    
    rsi_crossover_oversold = (rsi > rsiOversold) & (rsi.shift(1) <= rsiOversold)
    rsi_crossunder_overbought = (rsi < rsiOverbought) & (rsi.shift(1) >= rsiOverbought)
    
    longCondition = doubleBottom & rsi_crossover_oversold
    shortCondition = doubleTop & rsi_crossunder_overbought
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i < 2:
            continue
        
        if pd.isna(rsi.iloc[i]):
            continue
        
        entry_price = df['close'].iloc[i]
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        if longCondition.iloc[i]:
            entries.append({
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
        
        if shortCondition.iloc[i]:
            entries.append({
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
    
    return entries