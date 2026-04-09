import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.
    """
    # Volume filter
    volume_sma9 = df['volume'].rolling(9).mean()
    volfilt = df['volume'] > volume_sma9 * 1.5
    
    # ATR (Wilder) - True Range first
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atrfilt = ((df['low'] - df['high'].shift(2) > atr/1.5) | (df['low'].shift(2) - df['high'] > atr/1.5))
    
    # Trend filter using SMA(close, 54)
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # Bullish FVG: low > high[2]
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    
    # Bearish FVG: high < low[2]
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    # Bullish imbalance (TopImbalance_Bway)
    bullish_imb = (df['low'].shift(2) <= df['open'].shift(1)) & \
                  (df['high'] >= df['close'].shift(1)) & \
                  (df['close'] < df['low'].shift(1))
    
    # Bearish imbalance (BottomInbalance_Bway)
    bearish_imb = (df['high'].shift(2) >= df['open'].shift(1)) & \
                  (df['low'] <= df['close'].shift(1)) & \
                  (df['close'] > df['high'].shift(1))
    
    # Long entry: bullish FVG with bullish imbalance
    long_cond = bfvg & bullish_imb
    
    # Short entry: bearish FVG with bearish imbalance
    short_cond = sfvg & bearish_imb
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i < 2:
            continue
        
        if pd.isna(df['close'].iloc[i]):
            continue
        
        if long_cond.iloc[i]:
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
        
        if short_cond.iloc[i]:
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
            trade_num += 1
    
    return entries