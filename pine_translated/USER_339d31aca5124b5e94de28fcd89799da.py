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
    
    # Helper function to compute Wilder's RSI
    def compute_wilder_rsi(series: pd.Series, length: int) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Helper function to compute Wilder's ATR
    def compute_wilder_atr(df: pd.DataFrame, length: int) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr
    
    # Time window checks (London time: 08:00-09:59 and 15:00-16:59)
    ts = df['time']
    dt = pd.to_datetime(ts, unit='s', utc=True)
    hour = dt.dt.hour
    minute = dt.dt.minute
    
    isWithinMorningWindow = ((hour == 8) | ((hour == 9) & (minute <= 59)))
    isWithinAfternoonWindow = ((hour == 15) | ((hour == 16) & (minute <= 59)))
    in_trading_window = isWithinMorningWindow | isWithinAfternoonWindow
    
    # Filters (inputs default to false, but compute for completeness)
    volfilt = df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5
    
    atr = compute_wilder_atr(df, 20) / 1.5
    atrfilt = ((df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr))
    
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # Bullish FVG condition (long entry)
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb & in_trading_window
    
    # Bearish FVG condition (short entry)
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts & in_trading_window
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(df['low'].iloc[i]) or pd.isna(df['high'].iloc[i]):
            continue
        
        direction = None
        if bfvg.iloc[i]:
            direction = 'long'
        elif sfvg.iloc[i]:
            direction = 'short'
        
        if direction:
            entry_price = df['close'].iloc[i]
            ts_val = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(ts_val / 1000, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts_val,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries