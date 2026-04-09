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
    
    n = len(df)
    if n < 10:
        return results
    
    open_vals = df['open'].values
    high_vals = df['high'].values
    low_vals = df['low'].values
    close_vals = df['close'].values
    volume_vals = df['volume'].values
    time_vals = df['time'].values
    
    vol_sma = pd.Series(volume_vals).rolling(9).mean()
    close_sma_loc = pd.Series(close_vals).rolling(54).mean()
    
    tr = pd.Series(index=range(n), dtype=float)
    for i in range(1, n):
        high_i = high_vals[i]
        low_i = low_vals[i]
        prev_close = close_vals[i-1]
        tr.iloc[i] = max(high_i - low_i, abs(high_i - prev_close), abs(low_i - prev_close))
    
    atr = tr.ewm(alpha=1.0/20, adjust=False).mean()
    
    time_series = pd.Series(time_vals)
    dt_series = pd.to_datetime(time_series, unit='s', utc=True)
    hours = dt_series.dt.hour
    minutes = dt_series.dt.minute
    total_minutes = hours * 60 + minutes
    
    london_morning_start = 7 * 60 + 45
    london_morning_end = 9 * 60 + 45
    london_afternoon_start = 14 * 60 + 45
    london_afternoon_end = 16 * 60 + 45
    
    within_morning = (total_minutes >= london_morning_start) & (total_minutes < london_morning_end)
    within_afternoon = (total_minutes >= london_afternoon_start) & (total_minutes < london_afternoon_end)
    in_trading_window = within_morning | within_afternoon
    
    volfilt = volume_vals[1:] > vol_sma.iloc[1:] * 1.5
    volfilt = volfilt.reindex(range(n)).fillna(False).astype(bool)
    volfilt.iloc[0] = False
    
    loc2 = close_sma_loc > close_sma_loc.shift(1)
    loc2 = loc2.reindex(range(n)).fillna(False)
    
    atrfilt = pd.Series(False, index=range(n))
    for i in range(2, n):
        cond = (low_vals[i] - high_vals[i-2] > atr.iloc[i]) or (low_vals[i-2] - high_vals[i] > atr.iloc[i])
        atrfilt.iloc[i] = cond
    
    locfiltb = loc2.copy()
    locfilts = ~loc2
    
    bfvg = pd.Series(False, index=range(n))
    sfvg = pd.Series(False, index=range(n))
    
    for i in range(2, n):
        bfvg.iloc[i] = (low_vals[i] > high_vals[i-2])
        sfvg.iloc[i] = (high_vals[i] < low_vals[i-2])
    
    long_condition = (bfvg & volfilt & atrfilt & locfiltb & in_trading_window)
    short_condition = (sfvg & volfilt & atrfilt & locfilts & in_trading_window)
    
    for i in range(2, n):
        if long_condition.iloc[i]:
            entry_ts = int(time_vals[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close_vals[i])
            results.append({
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
        elif short_condition.iloc[i]:
            entry_ts = int(time_vals[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close_vals[i])
            results.append({
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
    
    return results