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
    if len(df) < 20:
        return []

    # Convert time to datetime for resampling
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('datetime', inplace=True)

    # Resample to 4H
    ohlc4h = df['close'].resample('240min').ohlc()
    high4h = df['high'].resample('240min').max()
    low4h = df['low'].resample('240min').min()
    volume4h = df['volume'].resample('240min').sum()
    high4h = high4h.reindex(ohlc4h.index).ffill()
    low4h = low4h.reindex(ohlc4h.index).ffill()
    volume4h = volume4h.reindex(ohlc4h.index).ffill()
    open4h = ohlc4h['open']
    close4h = ohlc4h['close']

    high4h = high4h.dropna()
    low4h = low4h.dropna()
    volume4h = volume4h.dropna()
    open4h = open4h.dropna()
    close4h = close4h.dropna()

    min_len = min(len(high4h), len(low4h), len(volume4h), len(open4h), len(close4h))
    high4h = high4h.iloc[:min_len]
    low4h = low4h.iloc[:min_len]
    volume4h = volume4h.iloc[:min_len]
    open4h = open4h.iloc[:min_len]
    close4h = close4h.iloc[:min_len]

    high4h.index = range(len(high4h))
    low4h.index = range(len(low4h))
    volume4h.index = range(len(volume4h))
    open4h.index = range(len(open4h))
    close4h.index = range(len(close4h))

    # Volume Filter - disabled by default (inp11 = false)
    volfilt1 = volume4h.shift(1) > volume4h.rolling(9).mean() * 1.5

    # ATR Filter - disabled by default (inp21 = false)
    atr_length1 = 20
    tr4h = pd.concat([high4h - low4h, (high4h - close4h.shift(1)).abs(), (low4h - close4h.shift(1)).abs()], axis=1).max(axis=1)
    atr4h = tr4h.ewm(alpha=1.0/atr_length1, adjust=False).mean() / 1.5
    atrfilt1 = (low4h - high4h.shift(2) > atr4h) | (low4h.shift(2) - high4h > atr4h)

    # Trend Filter - disabled by default (inp31 = false)
    loc1 = close4h.rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    locfiltb1 = loc21
    locfilts1 = ~loc21

    # Bullish and Bearish FVGs using 4H data
    bfvg1 = (low4h > high4h.shift(2)) & volfilt1 & atrfilt1 & locfiltb1
    sfvg1 = (high4h < low4h.shift(2)) & volfilt1 & atrfilt1 & locfilts1

    # Track last FVG type
    lastFVG = pd.Series(0, index=bfvg1.index)
    entries = []
    trade_num = 1

    confirmed_mask = pd.Series(True, index=bfvg1.index)
    new_4h_mask = pd.Series(True, index=bfvg1.index)
    new_4h_mask[1:] = bfvg1.index[1:] != bfvg1.index[:-1]
    new_4h_mask = new_4h_mask.cumsum()
    new_4h_mask = new_4h_mask.duplicated(keep='first')

    for i in range(2, len(bfvg1)):
        if confirmed_mask.iloc[i] and not new_4h_mask.iloc[i]:
            if bfvg1.iloc[i] and lastFVG.iloc[i-1] == -1:
                ts = int(close4h.index[i].timestamp())
                entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': entry_time,
                    'entry_price_guess': close4h.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close4h.iloc[i],
                    'raw_price_b': close4h.iloc[i]
                })
                trade_num += 1
                lastFVG.iloc[i] = 1
            elif sfvg1.iloc[i] and lastFVG.iloc[i-1] == 1:
                ts = int(close4h.index[i].timestamp())
                entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': entry_time,
                    'entry_price_guess': close4h.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close4h.iloc[i],
                    'raw_price_b': close4h.iloc[i]
                })
                trade_num += 1
                lastFVG.iloc[i] = -1
            elif bfvg1.iloc[i]:
                lastFVG.iloc[i] = 1
            elif sfvg1.iloc[i]:
                lastFVG.iloc[i] = -1
            else:
                lastFVG.iloc[i] = lastFVG.iloc[i-1]
        else:
            lastFVG.iloc[i] = lastFVG.iloc[i-1]

    return entries