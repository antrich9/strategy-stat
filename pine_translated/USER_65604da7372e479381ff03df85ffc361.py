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
    fvgTH = 0.5
    
    dt_index = pd.to_datetime(df['time'], unit='s', utc=True)
    dt_index = dt_index.tz_convert('Europe/London')
    hours = dt_index.hour
    minutes = dt_index.minute
    total_minutes = hours * 60 + minutes
    
    london_morning_start = 6 * 60 + 45
    london_morning_end = 9 * 60 + 45
    london_afternoon_start = 14 * 60 + 45
    london_afternoon_end = 16 * 60 + 45
    
    isWithinMorningWindow = (total_minutes >= london_morning_start) & (total_minutes < london_morning_end)
    isWithinAfternoonWindow = (total_minutes >= london_afternoon_start) & (total_minutes < london_afternoon_end)
    in_trading_window = isWithinMorningWindow | isWithinAfternoonWindow
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = pd.Series(np.nan, index=df.index)
    period = 144
    atr.iloc[period - 1] = tr.iloc[:period].mean()
    alpha = 2 / (period + 1)
    for i in range(period, len(tr)):
        atr.iloc[i] = atr.iloc[i - 1] * (1 - alpha) + tr.iloc[i] * alpha
    
    atr_filtered = atr * fvgTH
    
    bullG = low > high.shift(1)
    bearG = high < low.shift(1)
    
    bull = pd.Series(False, index=df.index)
    bear = pd.Series(False, index=df.index)
    
    valid_idx = df.index[period:]
    
    for i in valid_idx:
        atr_val = atr_filtered.iloc[i]
        if pd.isna(atr_val):
            continue
        
        bull.iloc[i] = (low.iloc[i] - high.iloc[i - 2]) > atr_val and \
                       low.iloc[i] > high.iloc[i - 2] and \
                       close.iloc[i - 1] > high.iloc[i - 2] and \
                       not (bullG.iloc[i] or bullG.iloc[i - 1])
        
        bear.iloc[i] = (low.iloc[i - 2] - high.iloc[i]) > atr_val and \
                       high.iloc[i] < low.iloc[i - 2] and \
                       close.iloc[i - 1] < low.iloc[i - 2] and \
                       not (bearG.iloc[i] or bearG.iloc[i - 1])
    
    entries = []
    trade_num = 0
    
    for i in range(len(df)):
        if not in_trading_window.iloc[i]:
            continue
        
        if bull.iloc[i] or bear.iloc[i]:
            ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = close.iloc[i]
            
            trade_num += 1
            direction = 'long' if bull.iloc[i] else 'short'
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
    
    return entries