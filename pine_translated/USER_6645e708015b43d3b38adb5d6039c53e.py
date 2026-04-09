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
    
    # Supertrend parameters
    atrPeriod = 10
    atrMultiplier = 3.0
    changeATR = True
    
    # ATR calculation (Wilder ATR)
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - df['close'].shift(1))
    tr3 = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/atrPeriod, adjust=False).mean()
    
    # Supertrend calculation
    src = (df['high'] + df['low']) / 2
    
    up = src - atrMultiplier * atr
    up = up.fillna(method='ffill')
    
    dn = src + atrMultiplier * atr
    dn = dn.fillna(method='ffill')
    
    # Trend calculation
    trend = pd.Series(1, index=df.index)
    trend.iloc[0] = 1
    
    for i in range(1, len(df)):
        prev_close = df['close'].iloc[i-1]
        prev_trend = trend.iloc[i-1]
        prev_up = up.iloc[i-1] if i > 0 else up.iloc[0]
        prev_dn = dn.iloc[i-1] if i > 0 else dn.iloc[0]
        
        if prev_trend == -1 and prev_close > prev_dn:
            trend.iloc[i] = 1
        elif prev_trend == 1 and prev_close < prev_up:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = prev_trend
    
    # Buy/Sell signals
    buySignal = (trend == 1) & (trend.shift(1) == -1)
    sellSignal = (trend == -1) & (trend.shift(1) == 1)
    
    # EMA calculations (Daily - using close as proxy since we don't have daily resampled data)
    ema8 = df['close'].ewm(span=8, adjust=False).mean()
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    
    # Long entry: buySignal + EMA alignment (ema8 > ema20 > ema50)
    long_condition = buySignal & (ema8 > ema20) & (ema20 > ema50)
    
    # Short entry: sellSignal + EMA alignment (ema8 < ema20 < ema50)
    short_condition = sellSignal & (ema8 < ema20) & (ema20 < ema50)
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(df['close'].iloc[i]):
            continue
            
        direction = None
        if long_condition.iloc[i]:
            direction = 'long'
        elif short_condition.iloc[i]:
            direction = 'short'
        
        if direction is not None:
            entry_price = df['close'].iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
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