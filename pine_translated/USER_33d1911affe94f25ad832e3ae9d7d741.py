import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Calculate MACD components
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macdLine = ema12 - ema26
    signalLine = macdLine.ewm(span=9, adjust=False).mean()
    
    # Bullish Engulfing Pattern
    bullishEngulfing = (
        (df['close'] > df['open']) &
        (df['close'].shift(1) < df['open'].shift(1)) &
        (df['close'] > df['open'].shift(1)) &
        (df['open'] < df['close'].shift(1))
    )
    
    # Bearish Engulfing Pattern
    bearishEngulfing = (
        (df['open'] > df['close']) &
        (df['open'].shift(1) < df['close'].shift(1)) &
        (df['open'] > df['close'].shift(1)) &
        (df['close'] < df['open'].shift(1))
    )
    
    # MACD crossover (macdLine crosses above signalLine)
    macdCrossover = (macdLine > signalLine) & (macdLine.shift(1) <= signalLine.shift(1))
    
    # MACD crossunder (macdLine crosses below signalLine)
    macdCrossunder = (macdLine < signalLine) & (macdLine.shift(1) >= signalLine.shift(1))
    
    # Buy and Sell conditions
    buyCondition = bullishEngulfing & macdCrossover
    sellCondition = bearishEngulfing & macdCrossunder
    
    entries = []
    trade_num = 0
    
    for i in range(1, len(df)):
        if pd.isna(macdLine.iloc[i]) or pd.isna(signalLine.iloc[i]):
            continue
        
        if buyCondition.iloc[i]:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
        
        if sellCondition.iloc[i]:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
    
    return entries