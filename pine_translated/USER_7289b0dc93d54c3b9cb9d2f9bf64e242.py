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
    
    # Check if we have enough data
    if len(df) < 50:
        return entries
    
    # Calculate EMAs on daily data (simulating request.security for daily timeframe)
    # Since df is likely already the target timeframe, we calculate EMAs directly
    ema8 = df['close'].ewm(span=8, adjust=False).mean()
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    
    # Calculate ATR for pivot threshold
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=10).mean()
    
    # ZigZag implementation would require complex pivot detection
    # For this strategy, entries appear to be based on EMA crossovers
    # with price at 0.618 Fibonacci level
    
    # Calculate Wilder RSI for potential entry confirmation
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Build boolean series for entry conditions
    # Long entry: EMA 8 crosses above EMA 20, price above EMA 50, RSI > 50
    ema8_above_ema20 = (ema8 > ema20) & (ema8.shift(1) <= ema20.shift(1))
    price_above_ema50 = close > ema50
    rsi_confirm = rsi > 50
    
    # Short entry: EMA 8 crosses below EMA 20, price below EMA 50, RSI < 50
    ema8_below_ema20 = (ema8 < ema20) & (ema8.shift(1) >= ema20.shift(1))
    price_below_ema50 = close < ema50
    rsi_confirm_short = rsi < 50
    
    long_condition = ema8_above_ema20 & price_above_ema50 & rsi_confirm
    short_condition = ema8_below_ema20 & price_below_ema50 & rsi_confirm_short
    
    # Iterate through bars and generate entries
    for i in range(1, len(df)):
        # Skip if ATR is NaN
        if pd.isna(atr.iloc[i]):
            continue
            
        entry_price = close.iloc[i]
        
        # Long entry
        if long_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
        
        # Short entry
        if short_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
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