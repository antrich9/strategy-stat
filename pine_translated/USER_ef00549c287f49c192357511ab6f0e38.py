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
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    bb = 20
    input_retSince = 2
    input_retValid = 2
    atrLength = 14
    atrMultiplier = 1.5
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    pl = low.rolling(2*bb+1, center=True).min().shift(bb).ffill()
    ph = high.rolling(2*bb+1, center=True).max().shift(bb).ffill()
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atrLength, adjust=False).mean()
    
    sTop = pl
    sBot = pl
    rTop = ph
    rBot = ph
    
    co = close > rTop
    co_prev = close.shift(1) <= rTop.shift(1)
    co_series = co & co_prev
    
    cu = close < sBot
    cu_prev = close.shift(1) >= sBot.shift(1)
    cu_series = cu & cu_prev
    
    df['rBreak'] = co_series
    df['sBreak'] = cu_series
    
    bars_since_rBreak = (df['rBreak'].cumsum() - (df['rBreak'].astype(int)).cumsum())
    bars_since_sBreak = (df['sBreak'].cumsum() - (df['sBreak'].astype(int)).cumsum())
    
    rRetActive = (bars_since_rBreak > input_retSince) & (high >= rTop) & (low <= rTop) & (low >= rBot) & (close > rTop)
    rRetValid = rRetActive & (bars_since_rBreak > 0) & (bars_since_rBreak <= input_retValid) & (close >= rTop) & (~rRetActive.shift(1).fillna(False))
    
    sRetActive = (bars_since_sBreak > input_retSince) & (low <= sBot) & (high >= sBot) & (high <= sTop) & (close < sBot)
    sRetValid = sRetActive & (bars_since_sBreak > 0) & (bars_since_sBreak <= input_retValid) & (close <= sBot) & (~sRetActive.shift(1).fillna(False))
    
    df['rRetValid'] = rRetValid
    df['sRetValid'] = sRetValid
    
    long_entry = df['rBreak'] & rRetActive
    short_entry = df['sBreak'] & sRetActive
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = close.iloc[i]
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
        
        if short_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = close.iloc[i]
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
    
    return entries