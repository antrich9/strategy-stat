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
    df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    ohlc_4h = df.set_index('timestamp').resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'time': 'first'
    }).dropna()
    
    high_4h = ohlc_4h['high']
    low_4h = ohlc_4h['low']
    close_4h = ohlc_4h['close']
    volume_4h = ohlc_4h['volume']
    time_4h = ohlc_4h['time']
    
    volfilt = volume_4h > volume_4h.rolling(9).mean() * 1.5
    
    high_low = high_4h - low_4h
    high_close_prev = np.abs(high_4h - close_4h.shift(1))
    low_close_prev = np.abs(low_4h - close_4h.shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr_4h = true_range.ewm(alpha=1/20, adjust=False).mean() / 1.5
    atrfilt = (low_4h - high_4h.shift(2) > atr_4h) | (low_4h.shift(2) - high_4h > atr_4h)
    
    loc = close_4h.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    bull_fvg = (low_4h > high_4h.shift(2)) & volfilt & atrfilt & locfiltb
    bear_fvg = (high_4h < low_4h.shift(2)) & volfilt & atrfilt & locfilts
    
    last_fvg = pd.Series(0, index=bull_fvg.index)
    bull_sharp = pd.Series(False, index=bull_fvg.index)
    bear_sharp = pd.Series(False, index=bear_fvg.index)
    
    for i in range(2, len(bull_fvg)):
        if bull_fvg.iloc[i]:
            if last_fvg.iloc[i-1] == -1:
                bull_sharp.iloc[i] = True
            last_fvg.iloc[i] = 1
        elif bear_fvg.iloc[i]:
            if last_fvg.iloc[i-1] == 1:
                bear_sharp.iloc[i] = True
            last_fvg.iloc[i] = -1
        else:
            last_fvg.iloc[i] = last_fvg.iloc[i-1]
    
    valid_bars = ~(bull_fvg.isna() | bear_fvg.isna() | loc.isna() | volfilt.isna() | atr_4h.isna())
    bull_sharp = bull_sharp & valid_bars
    bear_sharp = bear_sharp & valid_bars
    
    entries = []
    trade_num = 1
    
    for i in range(len(ohlc_4h)):
        if bull_sharp.iloc[i]:
            ts = int(time_4h.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close_4h.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(low_4h.iloc[i]),
                'raw_price_b': float(high_4h.shift(2).iloc[i])
            })
            trade_num += 1
        elif bear_sharp.iloc[i]:
            ts = int(time_4h.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close_4h.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(high_4h.iloc[i]),
                'raw_price_b': float(low_4h.shift(2).iloc[i])
            })
            trade_num += 1
    
    return entries