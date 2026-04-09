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
    entries = []
    trade_num = 1
    
    # Time window filters
    df['hour'] = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).hour)
    df['minute'] = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).minute)
    df['dayofweek'] = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).weekday())
    
    london_morning = ((df['hour'] == 8) & (df['minute'] >= 0)) | ((df['hour'] == 9) & (df['minute'] < 55))
    london_afternoon = ((df['hour'] == 14) & (df['minute'] >= 0)) | ((df['hour'] == 16) & (df['minute'] < 55))
    in_trading_window = london_morning | london_afternoon
    
    friday_morning = (df['dayofweek'] == 4) & ((df['hour'] == 8) & (df['minute'] >= 0))
    
    # Volume filter: inp1
    volfilt = df['volume'].shift(1) > df['volume'].rolling(9).mean().shift(1) * 1.5
    
    # ATR filter: inp2
    atr_atr = df['high'].rolling(14).max() - df['low'].rolling(14).min()
    atr_atr = atr_atr.where(atr_atr > 0, df['high'] - df['low'])
    atr_atr = atr_atr.ewm(alpha=1/14, adjust=False).mean()
    atr_filter_val = atr_atr / 1.5
    low_minus_high2 = df['low'] - df['high'].shift(2)
    low2_minus_high = df['low'].shift(2) - df['high']
    atrfilt = (low_minus_high2 > atr_filter_val) | (low2_minus_high > atr_filter_val)
    
    # Trend filter: inp3
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # Bullish FVG: low > high[2] and volfilt and atrfilt and locfiltb
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    
    # Bearish FVG: high < low[2] and volfilt and atrfilt and locfilts
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    # Long entries: bull FVG within trading window
    long_condition = bfvg & in_trading_window & ~friday_morning
    
    # Short entries: bear FVG within trading window
    short_condition = sfvg & in_trading_window & ~friday_morning
    
    for i in range(len(df)):
        if pd.isna(df['low'].iloc[i]) or pd.isna(df['high'].iloc[i]):
            continue
        if long_condition.iloc[i]:
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
    
    return entries