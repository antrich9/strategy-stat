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
    
    # Helper function for Wilder RSI
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    # Helper function for Wilder ATR
    def wilder_atr(high, low, close, length):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        return atr
    
    # Helper function for crossover
    def crossover(a, b, i):
        if i == 0:
            return False
        return a.iloc[i] > b.iloc[i] and a.iloc[i-1] <= b.iloc[i-1]
    
    # Helper function for crossunder
    def crossunder(a, b, i):
        if i == 0:
            return False
        return a.iloc[i] < b.iloc[i] and a.iloc[i-1] >= b.iloc[i-1]
    
    # Trading windows (London time, BST aware)
    def is_in_window(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        total_min = hour * 60 + minute
        # 07:00-11:45 (420-705)
        window1 = 420 <= total_min < 705
        # 14:00-14:45 (840-885)
        window2 = 840 <= total_min < 885
        return window1 or window2
    
    # Resample to 4H
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df_4h = df.set_index('datetime').resample('240min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    df_4h = df_4h.reset_index()
    
    close_4h = df_4h['close']
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    volume_4h = df_4h['volume']
    
    # 4H indicators
    loc1 = close_4h.ewm(span=54, adjust=False).mean()
    loc21 = loc1 > loc1.shift(1)
    volfilt1 = volume_4h.shift(1) > volume_4h.rolling(9).mean() * 1.5
    atr_4h = wilder_atr(high_4h, low_4h, close_4h, 20) / 1.5
    atrfilt1 = (low_4h - high_4h.shift(2) > atr_4h) | (low_4h.shift(2) - high_4h > atr_4h)
    locfiltb1 = loc21
    locfilts1 = ~loc21
    
    bfvg1 = (low_4h > high_4h.shift(2)) & volfilt1 & atrfilt1 & locfiltb1
    sfvg1 = (high_4h < low_4h.shift(2)) & volfilt1 & atrfilt1 & locfilts1
    
    # Chart timeframe indicators
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    volfilt = volume.shift(1) > volume.rolling(9).mean() * 1.5
    atr2 = wilder_atr(high, low, close, 20) / 1.5
    atrfilt = (low - high.shift(2) > atr2) | (low.shift(2) - high > atr2)
    loc = close.ewm(span=54, adjust=False).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    bfvg = (low > high.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (high < low.shift(2)) & volfilt & atrfilt & locfilts
    
    # State tracking
    lastFVG = 0
    prev_lastFVG = 0
    
    for i in range(len(df)):
        ts = int(df['time'].iloc[i])
        in_window = is_in_window(ts)
        
        # Update FVG state
        if i >= 2:
            if bfvg.iloc[i]:
                lastFVG = 1
            elif sfvg.iloc[i]:
                lastFVG = -1
        
        # Entry logic: Sharp Turn within window
        if in_window and i >= 2:
            entry_triggered = False
            direction = None
            
            if bfvg1.iloc[i] and prev_lastFVG == -1:
                entry_triggered = True
                direction = 'long'
            elif sfvg1.iloc[i] and prev_lastFVG == 1:
                entry_triggered = True
                direction = 'short'
            
            if entry_triggered:
                trade_num += 1
                entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                entry_price = float(close.iloc[i])
                
                entries.append({
                    'trade_num': trade_num,
                    'direction': direction,
                    'entry_ts': ts,
                    'entry_time': entry_time_str,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
        
        prev_lastFVG = lastFVG
    
    return entries