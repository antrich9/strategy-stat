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
    # Convert timestamp to hour for time filtering
    df = df.copy()
    df['hour'] = pd.to_datetime(df['time'], unit='s').dt.hour
    
    # Time condition: (hour >= 2 and hour < 5) or (hour >= 10 and hour < 12)
    isValidTradeTime = ((df['hour'] >= 2) & (df['hour'] < 5)) | ((df['hour'] >= 10) & (df['hour'] < 12))
    
    # ATR calculation (Wilder)
    atrLength = 14
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Wilder ATR
    atr = true_range.ewm(alpha=1.0/atrLength, adjust=False).mean()
    
    # Pivot Period
    PP = 5
    
    # Calculate pivots manually (simplified for major swings)
    # For a proper implementation, we'd need to look back and forward
    # Here we use a simplified version based on rolling max/min
    pivot_high = df['high'].rolling(window=PP*2+1, center=True).max() == df['high']
    pivot_low = df['low'].rolling(window=PP*2+1, center=True).min() == df['low']
    
    # For major pivots, we need higher timeframe logic - simplified here
    # Using rolling max of highs and min of lows as proxy
    major_high = df['high'].rolling(window=20).max()
    major_low = df['low'].rolling(window=20).min()
    
    # BB-like calculation (similar to Bollinger Bands concept mentioned in colors)
    bb_length = 20
    bb_mult = 2.0
    sma = df['close'].rolling(bb_length).mean()
    std = df['close'].rolling(bb_length).std()
    bb_upper = sma + bb_mult * std
    bb_lower = sma - bb_mult * std
    
    # Premium/Discount Zone (simplified PD concept)
    # Price in relation to recent range
    pd_range_high = df['high'].rolling(50).max()
    pd_range_low = df['low'].rolling(50).min()
    pd_mid = (pd_range_high + pd_range_low) / 2
    
    # Premium Zone: price > upper part of range
    # Discount Zone: price < lower part of range
    premium_zone = df['close'] > pd_mid + (pd_range_high - pd_mid) * 0.618
    discount_zone = df['close'] < pd_mid - (pd_range_low - pd_mid) * 0.618
    
    # Bullish/Bearish BB conditions (from variable names)
    # When price breaks out of BB in certain conditions
    bullish_bb = (df['close'] > bb_upper) & premium_zone
    bearish_bb = (df['close'] < bb_lower) & discount_zone
    
    # Crossover/Under detection for entry signals
    # Bullish crossover: close crosses above major resistance or BB upper
    prev_close_above_bb_upper = df['close'].shift(1) <= bb_upper.shift(1)
    bullish_cross = (df['close'] > bb_upper) & prev_close_above_bb_upper
    
    # Bearish crossunder: close crosses below major support or BB lower
    prev_close_below_bb_lower = df['close'].shift(1) >= bb_lower.shift(1)
    bearish_cross = (df['close'] < bb_lower) & prev_close_below_bb_lower
    
    # Entry conditions
    # Long: bullish signal + valid time
    long_condition = bullish_cross & isValidTradeTime
    
    # Short: bearish signal + valid time
    short_condition = bearish_cross & isValidTradeTime
    
    # Generate entries
    entries = []
    trade_num = 1
    
    # Iterate through bars
    for i in range(1, len(df)):
        entry_price = df['close'].iloc[i]
        ts = df['time'].iloc[i]
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        if long_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
        
        elif short_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
    
    return entries