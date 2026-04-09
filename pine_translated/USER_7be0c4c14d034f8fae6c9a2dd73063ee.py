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
    
    df = df.copy()
    
    # Volume filter: volume[1] > ta.sma(volume, 9) * 1.5
    vol_sma = df['volume'].rolling(9).mean()
    df['volfilt'] = df['volume'].shift(1) > vol_sma * 1.5
    
    # ATR filter using Wilder's method
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    df['atr'] = atr / 1.5
    
    # ATR filter: (low - high[2] > atr) or (low[2] - high > atr)
    df['atrfilt_bull'] = (low - high.shift(2) > df['atr']) | (low.shift(2) - high > df['atr'])
    df['atrfilt_bear'] = (low - high.shift(2) > df['atr']) | (low.shift(2) - high > df['atr'])
    
    # Trend filter: ta.sma(close, 54) vs its previous value
    sma54 = close.rolling(54).mean()
    df['loc'] = sma54
    df['loc2'] = df['loc'] > df['loc'].shift(1)
    df['locfiltb'] = df['loc2']
    df['locfilts'] = ~df['loc2']
    
    # FVG conditions
    df['bfvg'] = (low > high.shift(2)) & df['volfilt'] & df['atrfilt_bull'] & df['locfiltb']
    df['sfvg'] = (high < low.shift(2)) & df['volfilt'] & df['atrfilt_bear'] & df['locfilts']
    
    # Time filtering - London time windows (7:45-9:45 and 14:45-16:45), excluding Fridays
    dt = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_convert('Europe/London')
    df['london_dt'] = dt
    df['hour'] = dt.dt.hour
    df['minute'] = dt.dt.minute
    df['dayofweek'] = dt.dt.dayofweek  # Monday=0, Friday=4
    
    london_morning_start = (df['hour'] == 7) & (df['minute'] >= 45)
    london_morning_end = (df['hour'] == 9) & (df['minute'] <= 44)
    london_afternoon_start = (df['hour'] == 14) & (df['minute'] >= 45)
    london_afternoon_end = (df['hour'] == 16) & (df['minute'] <= 44)
    
    in_morning = (df['hour'] > 7) | ((df['hour'] == 7) & (df['minute'] >= 45))
    in_morning &= (df['hour'] < 9) | ((df['hour'] == 9) & (df['minute'] <= 44))
    
    in_afternoon = (df['hour'] > 14) | ((df['hour'] == 14) & (df['minute'] >= 45))
    in_afternoon &= (df['hour'] < 16) | ((df['hour'] == 16) & (df['minute'] <= 44))
    
    df['in_time_window'] = (in_morning | in_afternoon) & (df['dayofweek'] != 4)
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if df['in_time_window'].iloc[i]:
            if df['bfvg'].iloc[i]:
                ts = int(df['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': df['close'].iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': df['close'].iloc[i],
                    'raw_price_b': df['close'].iloc[i]
                })
                trade_num += 1
            elif df['sfvg'].iloc[i]:
                ts = int(df['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': df['close'].iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': df['close'].iloc[i],
                    'raw_price_b': df['close'].iloc[i]
                })
                trade_num += 1
    
    return entries