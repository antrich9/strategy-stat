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
    
    # ATR (Wilder) with period 20
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean() / 1.5
    
    # Volume filter
    volfilt = df['volume'] > df['volume'].rolling(9).mean() * 1.5
    
    # Trend filter
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # Bullish FVG: low > high[2]
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & (tr > atr) & locfiltb
    
    # Bearish FVG: high < low[2]
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & (tr > atr) & locfilts
    
    # Helper series for OB and FVG detection
    isUp = df['close'] > df['open']
    isDown = df['close'] < df['open']
    
    # Order Block conditions
    # Bullish OB: down bar at index+1, up bar at index, close > high[index+1]
    obUp = isDown.shift(1) & isUp & (df['close'] > df['high'].shift(1))
    
    # Bearish OB: up bar at index+1, down bar at index, close < low[index+1]
    obDown = isUp.shift(1) & isDown & (df['close'] < df['low'].shift(1))
    
    # FVG conditions
    fvgUp = df['low'] > df['high'].shift(2)
    fvgDown = df['high'] < df['low'].shift(2)
    
    # Trading windows (UTC): 07:00-10:59 and 15:00-16:59
    start_hour_1 = 7
    end_hour_1 = 10
    end_minute_1 = 59
    start_hour_2 = 15
    end_hour_2 = 16
    end_minute_2 = 59
    
    in_trading_window = pd.Series(False, index=df.index)
    
    # Entry conditions
    long_cond = bfvg & obUp & fvgUp
    short_cond = sfvg & obDown & fvgDown
    
    entries = []
    trade_num = 1
    
    for i in range(2, len(df)):
        ts = df['time'].iloc[i]
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        
        # Check trading windows
        in_win1 = (hour >= start_hour_1 and hour < end_hour_1) or (hour == end_hour_1 and minute <= end_minute_1)
        in_win2 = (hour >= start_hour_2 and hour < end_hour_2) or (hour == end_hour_2 and minute <= end_minute_2)
        in_window = in_win1 or in_win2
        
        if in_window:
            if long_cond.iloc[i]:
                entry_ts = int(ts)
                entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
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
            
            elif short_cond.iloc[i]:
                entry_ts = int(ts)
                entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
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