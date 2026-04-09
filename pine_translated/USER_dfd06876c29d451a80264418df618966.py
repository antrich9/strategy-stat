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
    results = []
    trade_num = 1
    
    # Calculate indicators
    volume_sma = df['volume'].rolling(9).mean()
    close_sma = df['close'].rolling(54).mean()
    
    # ATR (Wilder)
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    atr2 = atr / 1.5
    
    # Filters
    volfilt = df['volume'].shift(1) > volume_sma * 1.5
    atrfilt = (low - high.shift(2) > atr2) | (low.shift(2) - high > atr2)
    loc = close_sma
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # FVG conditions (short-term: low > high[2] for bullish, high < low[2] for bearish)
    bfvg1 = (low > high.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg1 = (high < low.shift(2)) & volfilt & atrfilt & locfilts
    
    # Trading windows (London)
    # Convert timestamps to datetime for window calculation
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    df['dayofweek'] = df['dt'].dt.dayofweek
    
    # London windows: 07:45-09:45 and 14:45-16:45
    isWithinWindow1 = (
        ((df['hour'] == 7) & (df['minute'] >= 45)) |
        ((df['hour'] == 8)) |
        ((df['hour'] == 9) & (df['minute'] < 45))
    )
    isWithinWindow2 = (
        ((df['hour'] == 14) & (df['minute'] >= 45)) |
        ((df['hour'] == 15)) |
        ((df['hour'] == 16) & (df['minute'] < 45))
    )
    in_trading_window = isWithinWindow1 | isWithinWindow2
    
    # Entry conditions
    long_condition = bfvg1 & in_trading_window
    short_condition = sfvg1 & in_trading_window
    
    # Generate entries
    for i in range(len(df)):
        if i < 2:
            continue
        
        entry_price = df['close'].iloc[i]
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        if long_condition.iloc[i] and not pd.isna(df['close'].iloc[i]):
            results.append({
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
        
        if short_condition.iloc[i] and not pd.isna(df['close'].iloc[i]):
            results.append({
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
    
    return results