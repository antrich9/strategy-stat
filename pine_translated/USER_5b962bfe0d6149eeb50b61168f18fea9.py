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
    
    # Helper function to check if bar is up (close > open)
    def is_up(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]
    
    # Helper function to check if bar is down (close < open)
    def is_down(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]
    
    # Helper function to check for bullish OB (Order Block)
    def is_ob_up(idx):
        return (is_down(idx + 1) and is_up(idx) and 
                df['close'].iloc[idx] > df['high'].iloc[idx + 1])
    
    # Helper function to check for bearish OB
    def is_ob_down(idx):
        return (is_up(idx + 1) and is_down(idx) and 
                df['close'].iloc[idx] < df['low'].iloc[idx + 1])
    
    # Helper function to check for bullish FVG (Fair Value Gap)
    def is_fvg_up(idx):
        return df['low'].iloc[idx] > df['high'].iloc[idx + 2]
    
    # Helper function to check for bearish FVG
    def is_fvg_down(idx):
        return df['high'].iloc[idx] < df['low'].iloc[idx + 2]
    
    # Time window check (London time: 07:45-09:45 and 14:45-16:45)
    def is_within_time_window(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        # Morning window: 07:45 to 09:45
        in_morning = (hour == 7 and minute >= 45) or (8 <= hour < 9) or (hour == 9 and minute <= 45)
        # Afternoon window: 14:45 to 16:45
        in_afternoon = (hour == 14 and minute >= 45) or (15 <= hour < 16) or (hour == 16 and minute <= 45)
        return in_morning or in_afternoon
    
    # Calculate volume filter
    vol_sma = df['volume'].rolling(9).mean()
    volfilt = df['volume'].shift(1) > vol_sma * 1.5
    
    # Calculate ATR (Wilder's method)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean() / 1.5
    
    # ATR filter
    atrfilt = (df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr)
    
    # Trend filter (SMA 54)
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # Bullish FVG condition
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    
    # Bearish FVG condition
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    # Calculate OB conditions for all bars (need lookback of 2 bars)
    n = len(df)
    ob_up = pd.Series(False, index=df.index)
    ob_down = pd.Series(False, index=df.index)
    
    for i in range(2, n):
        try:
            ob_up.iloc[i] = is_ob_up(i - 1)
            ob_down.iloc[i] = is_ob_down(i - 1)
        except:
            pass
    
    # FVG conditions (current bar)
    fvg_up = is_fvg_up(0)
    fvg_down = is_fvg_down(0)
    
    # Build combined conditions
    in_trading_window = df['time'].apply(is_within_time_window)
    
    # Long entry: within time window AND OB up on bar[1] AND FVG up on bar[0]
    long_condition = in_trading_window & ob_up & fvg_up
    
    # Short entry: within time window AND OB down on bar[1] AND FVG down on bar[0]
    short_condition = in_trading_window & ob_down & fvg_down
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(2, n):
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = df['close'].iloc[i]
        
        if long_condition.iloc[i]:
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
        elif short_condition.iloc[i]:
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