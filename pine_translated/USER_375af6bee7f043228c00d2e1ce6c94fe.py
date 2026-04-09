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
    # Extract OHLCV data
    open_price = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Parameters from Pine Script
    PP = 5  # Pivot Period from inputs
    atrLength = 14  # ATR Length from later inputs
    atrMultiplier = 1.0
    atrLength_long = 55  # From earlier in script
    
    # Calculate True Range (Wilder's method components)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder ATR (exponential weighted moving average with alpha=1/length)
    atr_14 = tr.ewm(alpha=1/atrLength, adjust=False).mean()
    atr_55 = tr.ewm(alpha=1/atrLength_long, adjust=False).mean()
    
    # Calculate pivot highs and lows (ta.pivothigh and ta.pivotlow)
    # For pivot high: high at bar i is pivot if it's highest over PP bars ending at i
    # Using rolling max shifted by 1 to get previous bar's value (similar to Pine's lookback)
    pivot_high = high.rolling(window=PP).max().shift(1)
    pivot_low = low.rolling(window=PP).min().shift(1)
    
    # Additional indicator: Rolling max/min for confirmation
    high_5 = high.rolling(window=5).max()
    low_5 = low.rolling(window=5).min()
    
    # State tracking variables (simplified from Pine Script)
    # States: BBplus=0, signUP=1, cnclUP=2, LL1break=3, LL2break=4, etc.
    # Entry signal: when state transitions to signUP (1)
    
    # Entry conditions for "Bullish PDL + BB":
    # 1. Price breaks above pivot high (bullish breakout)
    # 2. ATR confirms strength (close - open > ATR threshold)
    # 3. In discount zone (price near recent low) or after retracement
    
    # Bullish breakout: close crosses above pivot high
    bullish_breakout = (close > pivot_high) & (close.shift(1) <= pivot_high.shift(1))
    
    # ATR confirmation: body size relative to ATR
    candle_body = close - open_price
    atr_confirm = candle_body > atr_14 * 0.3
    
    # Discount zone detection: price is near lower part of recent range
    price_range = high_5 - low_5
    discount_zone = (low < low_5 + price_range * 0.2)  # Near 5-period low
    
    # Combine conditions
    entry_signal = bullish_breakout & atr_confirm & discount_zone
    
    # Alternative simpler entry: close above pivot high with strong ATR
    simple_entry = (close > pivot_high) & (candle_body > atr_14 * 0.5)
    
    # Use simple entry if discount zone not available (NaN at start)
    entry_long = entry_signal.fillna(False) | simple_entry.fillna(False)
    
    # Build entries list
    entries = []
    trade_num = 1
    
    # Start index after indicators are valid
    start_idx = max(atrLength, PP, 5) + 1
    
    for i in range(start_idx, len(df)):
        if entry_long.iloc[i]:
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
            trade_num += 1
    
    return entries