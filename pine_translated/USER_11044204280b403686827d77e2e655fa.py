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
    high = df['high']
    low = df['low']
    close = df['close']
    open_prices = df['open']
    n = len(df)
    
    lenHL = 20
    wickFactor = 1.0
    useATRbuf = True
    atrLen = 14
    
    tick = 1.0
    if n > 1:
        price_diffs = close.diff().dropna().abs()
        if len(price_diffs) > 0:
            tick = max(price_diffs.min(), 1e-6)
    
    tr = pd.Series(np.nan, index=df.index)
    tr.iloc[1:] = high.iloc[1:] - low.iloc[1:]
    tr.iloc[1:] = pd.concat([high.iloc[1:] - close.iloc[:-1].values, 
                             close.iloc[:-1].values - low.iloc[1:]], axis=1).max(axis=1).values
    atr = pd.Series(np.nan, index=df.index)
    atr.iloc[atrLen - 1] = tr.iloc[:atrLen].sum()
    alpha = 1.0 / atrLen
    for i in range(atrLen, n):
        if not np.isnan(atr.iloc[i - 1]):
            atr.iloc[i] = tr.iloc[i] * alpha + atr.iloc[i - 1] * (1 - alpha)
    
    buf = atr * wickFactor
    
    hh = pd.Series(np.nan, index=df.index)
    ll = pd.Series(np.nan, index=df.index)
    for i in range(lenHL, n):
        hh.iloc[i] = high.iloc[i - lenHL:i].max()
        ll.iloc[i] = low.iloc[i - lenHL:i].min()
    
    bullSweep = pd.Series(False, index=df.index)
    bearSweep = pd.Series(False, index=df.index)
    bullSweep.iloc[lenHL:] = (low.iloc[lenHL:] < ll.iloc[lenHL:].values) & (close.iloc[lenHL:] > ll.iloc[lenHL:].values)
    bearSweep.iloc[lenHL:] = (high.iloc[lenHL:] > hh.iloc[lenHL:].values) & (close.iloc[lenHL:] < hh.iloc[lenHL:].values)
    
    barRange = high - low
    bodySize = (close - open_prices).abs()
    isRejection = pd.Series(False, index=df.index)
    valid_bars = barRange > 0
    isRejection[valid_bars] = bodySize[valid_bars] / barRange[valid_bars] <= 0.5
    
    bullSignal = bullSweep & isRejection
    bearSignal = bearSweep & isRejection
    
    entries = []
    trade_num = 1
    
    position_open = False
    current_direction = None
    
    for i in range(n):
        if np.isnan(atr.iloc[i]) or np.isnan(hh.iloc[i]) or np.isnan(ll.iloc[i]):
            continue
        
        if bullSignal.iloc[i] and not position_open:
            entry_price = high.iloc[i] + tick
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            position_open = True
            current_direction = 'long'
        elif bearSignal.iloc[i] and not position_open:
            entry_price = low.iloc[i] - tick
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            position_open = True
            current_direction = 'short'
    
    return entries