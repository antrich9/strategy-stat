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
    
    is_up = df['close'] > df['open']
    is_down = df['close'] < df['open']
    
    ob_up = is_down.shift(1) & is_up & (df['close'] > df['high'].shift(1))
    ob_down = is_up.shift(1) & is_down & (df['close'] < df['low'].shift(1))
    
    fvg_up = df['low'] > df['high'].shift(2)
    fvg_down = df['high'] < df['low'].shift(2)
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - (100 / (1 + avg_gain / avg_loss))
    
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    vol_avg = df['volume'].rolling(9).mean()
    volfilt = df['volume'] > vol_avg * 1.5
    
    loc = df['close'].rolling(54).mean()
    loc_up = loc > loc.shift(1)
    locfiltb = loc_up
    locfilts = ~loc_up
    
    atrfilt = ((df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr))
    
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    top_imbalance = (df['low'].shift(2) <= df['open'].shift(1)) & (df['high'] >= df['close'].shift(1)) & (df['close'] < df['low'].shift(1))
    top_imbalance_xbway = (df['low'].shift(2) <= df['open'].shift(1)) & (df['high'] >= df['close'].shift(1)) & (df['close'] > df['low'].shift(1))
    
    df_time = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_convert('Europe/London')
    df['hour'] = df_time.dt.hour
    df['minute'] = df_time.dt.minute
    
    morning_start = (df['hour'] == 8) & (df['minute'] >= 0)
    morning_end = (df['hour'] == 9) & (df['minute'] <= 45)
    in_morning_window = morning_start | morning_end
    
    afternoon_start = (df['hour'] == 15) & (df['minute'] >= 0)
    afternoon_end = (df['hour'] == 16) & (df['minute'] <= 45)
    in_afternoon_window = afternoon_start | afternoon_end
    
    in_trading_window = in_morning_window | in_afternoon_window
    
    long_condition = (bfvg | fvg_up) & ob_up & top_imbalance & in_trading_window
    short_condition = (sfvg | fvg_down) & ob_down & top_imbalance_xbway & in_trading_window
    
    for i in range(1, len(df)):
        if i == 0 or pd.isna(df['close'].iloc[i-1]):
            continue
        
        long_signal = long_condition.iloc[i]
        short_signal = short_condition.iloc[i]
        
        if long_signal or short_signal:
            direction = 'long' if long_signal else 'short'
            ts = int(df['time'].iloc[i])
            dt = pd.to_datetime(ts, unit='ms', utc=True).tz_convert('Europe/London')
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': dt.isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries