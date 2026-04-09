import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Input parameters from Pine Script
    ema_short_period = 8
    ema_medium_period = 20
    ema_long_period = 50
    rsi_length = 14
    atr_length = 14
    atr_threshold = 0.01
    
    # Calculate EMAs
    short_ema = df['close'].ewm(span=ema_short_period, adjust=False).mean()
    medium_ema = df['close'].ewm(span=ema_medium_period, adjust=False).mean()
    long_ema = df['close'].ewm(span=ema_long_period, adjust=False).mean()
    
    # Identify Market Phase
    uptrend = (short_ema > medium_ema) & (medium_ema > long_ema)
    downtrend = (short_ema < medium_ema) & (medium_ema < long_ema)
    
    # Calculate RSI using Wilder smoothing
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/rsi_length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi_value = 100 - (100 / (1 + rs))
    
    # Calculate ATR using Wilder smoothing
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_value = tr.ewm(alpha=1/atr_length, adjust=False).mean()
    
    # Volatility condition
    volatility_condition = atr_value > atr_threshold
    
    # Long Entry: crossover short EMA above medium EMA AND uptrend AND RSI > 50 AND volatility condition
    long_entry = (short_ema > medium_ema) & (short_ema.shift() <= medium_ema.shift()) & uptrend & (rsi_value > 50) & volatility_condition
    
    # Short Entry: crossunder short EMA below medium EMA AND downtrend AND RSI < 50 AND volatility condition
    short_entry = (short_ema < medium_ema) & (short_ema.shift() >= medium_ema.shift()) & downtrend & (rsi_value < 50) & volatility_condition
    
    # Build entries list
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
        entry_price = float(df['close'].iloc[i])
        
        if long_entry.iloc[i]:
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
        
        if short_entry.iloc[i]:
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
            trade_num += 1
    
    return entries