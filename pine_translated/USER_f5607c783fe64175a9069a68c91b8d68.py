import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    baselineLength = 50
    cciLength = 20
    atrLength = 14
    atrVolatilityThreshold = 1.5
    cciOversold = -100
    cciOverbought = 100
    
    baselineEMA = df['close'].ewm(span=baselineLength, adjust=False).mean()
    
    period = cciLength
    sma = df['close'].rolling(window=period).mean()
    mean_dev = df['close'].rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (df['close'] - sma) / (0.015 * mean_dev)
    cci = cci.replace([np.inf, -np.inf], np.nan)
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/atrLength, adjust=False).mean()
    
    close_series = df['close']
    
    longCondition = (close_series > baselineEMA) & (cci > cciOversold) & (atr > atrVolatilityThreshold)
    shortCondition = (close_series < baselineEMA) & (cci < cciOverbought) & (atr > atrVolatilityThreshold)
    
    longCrossover = (close_series > baselineEMA) & (close_series.shift(1) <= baselineEMA.shift(1))
    shortCrossunder = (close_series < baselineEMA) & (close_series.shift(1) >= baselineEMA.shift(1))
    
    entries = []
    trade_num = 1
    
    start_idx = max(baselineLength, cciLength, atrLength)
    
    for i in range(start_idx, len(df)):
        if longCrossover.iloc[i] and longCondition.iloc[i] and not np.isnan(baselineEMA.iloc[i]) and not np.isnan(cci.iloc[i]) and not np.isnan(atr.iloc[i]):
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': df['time'].iloc[i],
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        if shortCrossunder.iloc[i] and shortCondition.iloc[i] and not np.isnan(baselineEMA.iloc[i]) and not np.isnan(cci.iloc[i]) and not np.isnan(atr.iloc[i]):
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': df['time'].iloc[i],
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries