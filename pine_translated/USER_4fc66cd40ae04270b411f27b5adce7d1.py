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
    
    close = df['close']
    open_price = df['open']
    
    short_ema = close.ewm(span=12, adjust=False).mean()
    long_ema = close.ewm(span=26, adjust=False).mean()
    macdLine = short_ema - long_ema
    signalLine = macdLine.ewm(span=9, adjust=False).mean()
    
    bullishEngulfing = (close > open_price) & (close.shift(1) < open_price.shift(1)) & (close > open_price.shift(1)) & (open_price < close.shift(1))
    bearishEngulfing = (open_price > close) & (open_price.shift(1) < close.shift(1)) & (open_price > close.shift(1)) & (close < open_price.shift(1))
    
    macdCrossUp = (macdLine > signalLine) & (macdLine.shift(1) <= signalLine.shift(1))
    macdCrossDown = (macdLine < signalLine) & (macdLine.shift(1) >= signalLine.shift(1))
    
    buyCondition = bullishEngulfing & macdCrossUp
    sellCondition = bearishEngulfing & macdCrossDown
    
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if pd.isna(macdLine.iloc[i]) or pd.isna(signalLine.iloc[i]):
            continue
        if buyCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        elif sellCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
    
    return entries