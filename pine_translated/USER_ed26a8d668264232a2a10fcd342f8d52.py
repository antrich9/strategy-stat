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
    
    # Extract hour from timestamp for time filtering
    hours = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).hour)
    
    # Time filter: 02:00-05:00 or 10:00-12:00 UTC
    is_valid_time = ((hours >= 2) & (hours < 5)) | ((hours >= 10) & (hours < 12))
    
    # Compute Order Block (OB) conditions
    # isObDown: bearish candle followed by bullish candle, close > high of prev
    ob_down = (
        (df['close'].shift(1) < df['open'].shift(1)) &  # isDown(index + 1) -> prev bar bearish
        (df['close'] > df['open']) &  # isUp(index) -> current bar (relative to prev) bullish
        (df['close'] > df['high'].shift(1))  # close > high[index + 1]
    )
    
    # isObUp: bullish candle followed by bearish candle, close < low of prev
    ob_up = (
        (df['close'].shift(1) > df['open'].shift(1)) &  # isUp(index + 1) -> prev bar bullish
        (df['close'] < df['open']) &  # isDown(index) -> current bar (relative to prev) bearish
        (df['close'] < df['low'].shift(1))  # close < low[index + 1]
    )
    
    # Compute Fair Value Gap (FVG) conditions
    # isFvgUp: low > high[2] (bullish FVG / up-thrust)
    fvg_up = df['low'] > df['high'].shift(2)
    
    # isFvgDown: high < low[2] (bearish FVG / down-thrust)
    fvg_down = df['high'] < df['low'].shift(2)
    
    # Bullish entry: obDown AND fvgUp
    # obDown at i means: isDown(i) and isUp(i+1) and close(i+1) > high(i)
    # fvgUp at i means: low(i) > high(i-2)
    # Combined: bearish OB formed at previous bar, bullish FVG at current bar
    long_entries = ob_down & fvg_up & is_valid_time
    
    # Bearish entry: obUp AND fvgDown
    # obUp at i means: isUp(i) and isDown(i+1) and close(i+1) < low(i)
    # fvgDown at i means: high(i) < low(i-2)
    # Combined: bullish OB formed at previous bar, bearish FVG at current bar
    short_entries = ob_up & fvg_down & is_valid_time
    
    entries_list = []
    trade_num = 1
    
    for i in df.index:
        if long_entries.loc[i]:
            ts = int(df['time'].loc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            price = float(df['close'].loc[i])
            entries_list.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
    
    for i in df.index:
        if short_entries.loc[i]:
            ts = int(df['time'].loc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            price = float(df['close'].loc[i])
            entries_list.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
    
    return entries_list