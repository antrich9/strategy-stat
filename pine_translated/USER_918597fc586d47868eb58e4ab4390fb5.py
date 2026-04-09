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
    trade_num = 0
    
    # Parameters
    PP = 5  # Pivot Period
    atrLength = 14
    
    # Calculate pivots
    high_pivot = df['high'].rolling(window=PP+1, center=True).max() == df['high']
    low_pivot = df['low'].rolling(window=PP+1, center=True).min() == df['low']
    
    # Calculate ZigZag values
    # Use pivot-based approach
    pivots_high = np.where(high_pivot, df['high'], np.nan)
    pivots_low = np.where(low_pivot, df['low'], np.nan)
    
    # Forward fill to get last pivot values
    last_high_pivot = pd.Series(pivots_high).ffill()
    last_low_pivot = pd.Series(pivots_low).ffill()
    
    # Calculate ATR (Wilder)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Wilder ATR
    atr = true_range.ewm(alpha=1/atrLength, adjust=False).mean()
    
    # Calculate EMAs for trend
    ema_9 = df['close'].ewm(span=9, adjust=False).mean()
    ema_21 = df['close'].ewm(span=21, adjust=False).mean()
    
    # Calculate RSI (Wilder)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Detect swing highs and lows for structure
    # Major structure detection
    swing_high = (df['high'].shift(1) > df['high'].shift(2)) & (df['high'].shift(1) > df['high'])
    swing_low = (df['low'].shift(1) < df['low'].shift(2)) & (df['low'].shift(1) < df['low'])
    
    # Detect BoS and ChoCh conditions
    # For bullish: price breaking above previous high structure
    # For bearish: price breaking below previous low structure
    
    # Simple structure detection based on pivots
    # Track if we're making higher highs (bullish) or lower lows (bearish)
    major_highs = pd.Series(np.where(high_pivot, df['high'], np.nan)).ffill()
    major_lows = pd.Series(np.where(low_pivot, df['low'], np.nan)).ffill()
    
    # Calculate midpoints
    midpoints = (major_highs + major_lows) / 2
    
    # Detect structure breaks
    # Bullish BoS: price crosses above previous major high
    bullish_bos = df['close'] > major_highs.shift(1)
    
    # Bearish BoS: price crosses below previous major low
    bearish_bos = df['close'] < major_lows.shift(1)
    
    # Detect ChoCh (Change of Character) - more significant breaks
    # Bullish ChoCh: price breaks above with momentum (RSI > 50)
    bullish_choch = bullish_bos & (rsi > 50) & (ema_9 > ema_21)
    
    # Bearish ChoCh: price breaks below with momentum (RSI < 50)
    bearish_choch = bearish_bos & (rsi < 50) & (ema_9 < ema_21)
    
    # Entry conditions based on midpoint fib 0.5 (50% retracement)
    # Long entry: price pulls back to midpoint and bounces (structure break confirmed)
    midpoint_cross_up = (df['close'] > midpoints) & (df['close'].shift(1) <= midpoints.shift(1))
    midpoint_cross_down = (df['close'] < midpoints) & (df['close'].shift(1) >= midpoints.shift(1))
    
    # Combined entry signals
    # Long: bullish structure (BoS or ChoCh) + price at/below midpoint + confirmation
    long_entry = (bullish_bos | bullish_choch) & (df['close'] <= midpoints * 1.01) & (df['close'] > df['low'].shift(1))
    
    # Short: bearish structure (BoS or ChoCh) + price at/above midpoint + confirmation
    short_entry = (bearish_bos | bearish_choch) & (df['close'] >= midpoints * 0.99) & (df['close'] < df['high'].shift(1))
    
    # Filter out NaN values from indicators
    valid_bars = ~(midpoints.isna() | rsi.isna() | atr.isna())
    
    # State tracking
    is_long_open = False
    is_short_open = False
    
    for i in range(len(df)):
        if i < PP + 2:  # Skip bars where pivots might not be valid
            continue
        if pd.isna(midpoints.iloc[i]) or pd.isna(rsi.iloc[i]) or pd.isna(atr.iloc[i]):
            continue
        if not valid_bars.iloc[i]:
            continue
            
        # Check for entry conditions
        if long_entry.iloc[i] and not is_long_open:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
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
            is_long_open = True
            
        if short_entry.iloc[i] and not is_short_open:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
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
            is_short_open = True
    
    return entries