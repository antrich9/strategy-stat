import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Calculate EMAs
    fastEMA = df['close'].ewm(span=8, adjust=False).mean()
    mediumEMA = df['close'].ewm(span=20, adjust=False).mean()
    slowEMA = df['close'].ewm(span=50, adjust=False).mean()
    
    # Calculate price thresholds
    hl_range = df['high'] - df['low']
    upperThreshold = df['high'] - hl_range * 0.33
    lowerThreshold = df['low'] + hl_range * 0.33
    
    # Calculate Wilder RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Define candle conditions
    bullishCandle = (df['close'] > upperThreshold) & (df['open'] > upperThreshold) & (df['close'] > fastEMA)
    bearishCandle = (df['close'] < lowerThreshold) & (df['open'] < lowerThreshold) & (df['close'] < fastEMA)
    
    # EMA alignment conditions
    longEMAsAligned = (fastEMA > mediumEMA) & (mediumEMA > slowEMA)
    shortEMAsAligned = (fastEMA < mediumEMA) & (mediumEMA < slowEMA)
    
    # Entry conditions
    longCondition = bullishCandle & longEMAsAligned & (rsi < 70)
    shortCondition = bearishCandle & shortEMAsAligned & (rsi > 30)
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if (pd.isna(fastEMA.iloc[i]) or pd.isna(mediumEMA.iloc[i]) or 
            pd.isna(slowEMA.iloc[i]) or pd.isna(rsi.iloc[i])):
            continue
        
        if longCondition.iloc[i]:
            ts = df['time'].iloc[i]
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif shortCondition.iloc[i]:
            ts = df['time'].iloc[i]
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
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