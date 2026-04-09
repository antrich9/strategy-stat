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
    
    # Wilder RSI implementation
    def wilder_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Wilder ATR implementation
    def wilder_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    # Calculate indicators
    atr_length = 14
    atr_value = wilder_atr(df['high'], df['low'], df['close'], atr_length)
    
    pivot_period = 5  # PP = input.int(5, ...)
    
    # Zig Zag calculation
    high_pivot = df['high'].rolling(window=pivot_period+1).max().shift(pivot_period) == df['high']
    low_pivot = df['low'].rolling(window=pivot_period+1).min().shift(pivot_period) == df['low']
    
    # Detect swing highs and lows (simplified ZigZag)
    swing_high = (df['high'].shift(1) > df['high'].shift(2)) & (df['high'].shift(1) > df['high'])
    swing_low = (df['low'].shift(1) < df['low'].shift(2)) & (df['low'].shift(1) < df['low'])
    
    # Market Structure Detection - BOS and ChoCh
    # Major and Minor structure detection
    PP = pivot_period
    
    # Calculate pivot highs and lows
    high_pivots = df['high'].shift(-PP)
    low_pivots = df['low'].shift(-PP)
    
    # Detect BOS (Break of Structure) - simplified
    # Major trend detection using SMA
    sma_20 = df['close'].rolling(20).mean()
    sma_50 = df['close'].rolling(50).mean()
    
    # Major and Minor levels (simplified detection)
    # Looking for structure breaks
    major_high = df['high'].rolling(20).max()
    major_low = df['low'].rolling(20).min()
    minor_high = df['high'].rolling(10).max()
    minor_low = df['low'].rolling(10).min()
    
    # Structure flags
    major_bull_bos = (df['close'] > major_high.shift(1)) & (sma_20 > sma_50)
    major_bear_bos = (df['close'] < major_low.shift(1)) & (sma_20 < sma_50)
    
    # ChoCh (Change of Character) - momentum divergence
    rsi = wilder_rsi(df['close'], 14)
    rsi_divergence_bull = (rsi < 30) & (df['low'] < df['low'].shift(1))
    rsi_divergence_bear = (rsi > 70) & (df['high'] > df['high'].shift(1))
    
    # Entry conditions based on strategy name: "no engulf fib 0.782 s cont"
    # Look for Fibonacci 0.782 retracement levels
    # Major swing high/low for fib calculation
    swing_high_val = df['high'].rolling(20).max()
    swing_low_val = df['low'].rolling(20).min()
    swing_range = swing_high_val - swing_low_val
    
    # Fibonacci 0.782 retracement levels
    fib_0782_up = swing_low_val + swing_range * 0.782
    fib_0782_down = swing_high_val - swing_range * 0.782
    
    # Price approaching or at fib 0.782 with structure confirmation
    near_fib_0782_bull = (df['low'] <= fib_0782_up * 1.01) & (df['low'] >= fib_0782_up * 0.99)
    near_fib_0782_bear = (df['high'] >= fib_0782_down * 0.99) & (df['high'] <= fib_0782_down * 1.01)
    
    # Long Entry: DT (Double Top) pattern broken, bullish structure, at fib 0.782
    dt_bull_entry = (
        major_bull_bos | rsi_divergence_bull
    ) & near_fib_0782_bull & (
        df['close'] > df['open']  # Bullish candle
    )
    
    # Short Entry: DB (Double Bottom) pattern broken, bearish structure, at fib 0.782
    db_bear_entry = (
        major_bear_bos | rsi_divergence_bear
    ) & near_fib_0782_bear & (
        df['close'] < df['open']  # Bearish candle
    )
    
    # Filter for valid entries (skip NaN bars)
    valid_bull = dt_bull_entry & (~dt_bull_entry.isna()) & (atr_value > 0)
    valid_bear = db_bear_entry & (~db_bear_entry.isna()) & (atr_value > 0)
    
    entries = []
    trade_num = 1
    
    # Generate long entries
    for i in range(len(df)):
        if valid_bull.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    # Generate short entries
    for i in range(len(df)):
        if valid_bear.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
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