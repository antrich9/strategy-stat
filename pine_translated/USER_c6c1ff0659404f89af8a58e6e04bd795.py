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
    
    PP = 5
    
    # Calculate ATR (Wilder)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/14, min_periods=14).mean()
    
    # Pivot detection
    pivot_high = pd.Series(False, index=df.index)
    pivot_low = pd.Series(False, index=df.index)
    
    for i in range(PP, len(df) - PP):
        if df['high'].iloc[i] == df['high'].iloc[i_PP:i+PP+1].max():
            pivot_high.iloc[i] = True
        if df['low'].iloc[i] == df['low'].iloc[i_PP:i+PP+1].min():
            pivot_low.iloc[i] = True
    
    # Get pivot values
    pivot_high_val = pd.Series(np.where(pivot_high, df['high'], np.nan), index=df.index)
    pivot_low_val = pd.Series(np.where(pivot_low, df['low'], np.nan), index=df.index)
    
    # Forward fill pivot values
    pivot_high_val = pivot_high_val.ffill()
    pivot_low_val = pivot_low_val.ffill()
    
    # Market structure detection
    major_bullish_bos = pd.Series(False, index=df.index)
    major_bearish_bos = pd.Series(False, index=df.index)
    major_bullish_choch = pd.Series(False, index=df.index)
    major_bearish_choch = pd.Series(False, index=df.index)
    
    # Double Top/Bottom signals
    dt_signal = pd.Series(False, index=df.index)
    db_signal = pd.Series(False, index=df.index)
    
    last_highs = []
    last_lows = []
    
    for i in range(PP, len(df)):
        if pivot_high.iloc[i]:
            last_highs.append((i, df['high'].iloc[i]))
            if len(last_highs) > 3:
                last_highs.pop(0)
        
        if pivot_low.iloc[i]:
            last_lows.append((i, df['low'].iloc[i]))
            if len(last_lows) > 3:
                last_lows.pop(0)
        
        if len(last_highs) >= 2:
            h1, v1 = last_highs[-2]
            h2, v2 = last_highs[-1]
            if h2 > h1 and v2 > v1 and i == h2:
                dt_signal.iloc[i] = True
        
        if len(last_lows) >= 2:
            l1, v1 = last_lows[-2]
            l2, v2 = last_lows[-1]
            if l2 > l1 and v2 < v1 and i == l2:
                db_signal.iloc[i] = True
    
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)
    
    major_bullish_bos = (df['close'] > prev_high) & (df['close'] > df['open']) & (atr > atr.rolling(20).mean())
    major_bearish_bos = (df['close'] < prev_low) & (df['close'] < df['open']) & (atr > atr.rolling(20).mean())
    
    major_bullish_choch = (pivot_low_val > pivot_low_val.shift(PP).ffill()) & (df['close'] > df['open']) & (atr > atr.rolling(20).mean())
    major_bearish_choch = (pivot_high_val < pivot_high_val.shift(PP).ffill()) & (df['close'] < df['open']) & (atr > atr.rolling(20).mean())
    
    long_condition = (major_bullish_bos | major_bullish_choch) & dt_signal
    short_condition = (major_bearish_bos | major_bearish_choch) & db_signal
    
    entries = []
    trade_num = 1
    is_long_open = False
    is_short_open = False
    
    for i in range(len(df)):
        if i < 50:
            continue
        
        if long_condition.iloc[i] and not is_long_open:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            is_long_open = True
            is_short_open = False
        
        elif short_condition.iloc[i] and not is_short_open:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            is_short_open = True
            is_long_open = False
    
    return entries