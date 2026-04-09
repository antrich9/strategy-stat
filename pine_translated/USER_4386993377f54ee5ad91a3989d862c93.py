import pandas as pd
import numpy as np
from datetime import datetime, timezone

def calculate_wilder_rsi(series, length):
    alpha = 1.0 / length
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=alpha, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_wilder_atr(high, low, close, length):
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    return atr

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
    
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    isUp = close > open_
    isDown = close < open_
    
    obUp = isDown.shift(1) & isUp & (close > high.shift(1))
    obDown = isUp.shift(1) & isDown & (close < low.shift(1))
    
    fvgUp = low > high.shift(2)
    fvgDown = high < low.shift(2)
    
    volfilt = (volume > volume.rolling(9).mean() * 1.5)
    
    atr = calculate_wilder_atr(high, low, close, 20) / 1.5
    atrfilt = (low - high.shift(2) > atr) | (low.shift(2) - high > atr)
    
    loc = close.ewm(span=54, adjust=False).mean()
    locfiltb = loc > loc.shift(1)
    locfilts = ~locfiltb
    
    bfvg = fvgUp & volfilt & atrfilt & locfiltb
    sfvg = fvgDown & volfilt & atrfilt & locfilts
    
    longCondition = obUp & fvgUp & bfvg
    shortCondition = obDown & fvgDown & sfvg
    
    trade_num = 1
    
    for i in range(2, len(df)):
        ts = int(df['time'].iloc[i])
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        if longCondition.iloc[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': dt,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        elif shortCondition.iloc[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': dt,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
    
    return results