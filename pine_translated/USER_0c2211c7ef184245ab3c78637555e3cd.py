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
    time_col = df['time']
    
    # --- Indicator Calculations ---
    
    # 1. Wilder RSI (length 11)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing uses alpha = 1/length
    avg_gain = gain.ewm(alpha=1.0/11, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/11, adjust=False).mean()
    rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))

    # 2. TDI Components (lengths 31, 1, 9)
    bandLength = 31
    lengthrsipl = 1
    lengthtradesl = 9
    
    ma = rsi.rolling(bandLength).mean()
    std = rsi.rolling(bandLength).std() # Sample std (ddof=1), matches Pine default
    offs = 1.6185 * std
    up = ma + offs
    dn = ma - offs
    fastMA = rsi.rolling(lengthrsipl).mean()
    
    # 3. EMA and TEMA (length 9)
    lengthTEMA = 9
    lengthEMA = 9
    
    ema1 = close.ewm(span=lengthTEMA, adjust=False).mean()
    ema2 = ema1.ewm(span=lengthTEMA, adjust=False).mean()
    ema3 = ema2.ewm(span=lengthTEMA, adjust=False).mean()
    tema = 3 * (ema1 - ema2) + ema3
    ema = close.ewm(span=lengthEMA, adjust=False).mean()
    
    # Background Condition: close > ema AND close > tema
    bgCondition = (close > ema) & (close > tema)
    
    # --- State Variable Iteration ---
    entries = []
    trade_num = 0
    
    # Pine var initialization
    last_break_b