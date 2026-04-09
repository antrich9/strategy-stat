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
    
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    
    # London time windows
    london_start_morning = df['time'].dt.tz_convert('Europe/London').dt.normalize() + pd.Timedelta(hours=7, minutes=45)
    london_end_morning = df['time'].dt.tz_convert('Europe/London').dt.normalize() + pd.Timedelta(hours=9, minutes=45)
    london_start_afternoon = df['time'].dt.tz_convert('Europe/London').dt.normalize() + pd.Timedelta(hours=14, minutes=45)
    london_end_afternoon = df['time'].dt.tz_convert('Europe/London').dt.normalize() + pd.Timedelta(hours=16, minutes=45)
    
    isWithinMorningWindow = (df['time'] >= london_start_morning) & (df['time'] < london_end_morning)
    isWithinAfternoonWindow = (df['time'] >= london_start_afternoon) & (df['time'] < london_end_afternoon)
    in_trading_window = isWithinMorningWindow | isWithinAfternoonWindow
    
    # Filters
    volfilt = (df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5)
    loc = df['close'].rolling(54).mean()
    locfiltb = loc > loc.shift(1)
    locfilts = ~locfiltb
    
    # FVG conditions
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & locfiltb
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & locfilts
    
    # OB conditions
    obUp = (df['close'].shift(2) < df['open'].shift(2)) & (df['close'].shift(1) > df['open'].shift(1)) & (df['close'].shift(1) > df['high'].shift(2))
    obDown = (df['close'].shift(2) > df['open'].shift(2)) & (df['close'].shift(1) < df['open'].shift(1)) & (df['close'].shift(1) < df['low'].shift(2))
    
    # Entry signals
    long_entry = bfvg & obUp.shift(1) & in_trading_window
    short_entry = sfvg & obDown.shift(1) & in_trading_window
    
    # Iterate and generate entries
    for i in range(len(df)):
        if pd.isna(long_entry.iloc[i]) or pd.isna(short_entry.iloc[i]):
            continue
        
        if long_entry.iloc[i]:
            ts = int(df['time'].iloc[i].timestamp() * 1000)
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        
        if short_entry.iloc[i]:
            ts = int(df['time'].iloc[i].timestamp() * 1000)
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries