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
    if len(df) < 5:
        return []

    entries = []
    trade_num = 1

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    volfilt = volume.shift(1) > volume.rolling(9).mean() * 1.5
    atr = _calculate_atr(df, 20) / 1.5

    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    dailyHigh = high
    dailyLow = low
    dailyHigh1 = high.shift(1)
    dailyLow1 = low.shift(1)
    dailyHigh2 = high.shift(2)
    dailyLow2 = low.shift(2)
    prevDayHigh = high.shift(1)
    prevDayLow = low.shift(1)

    atrfilt = ((dailyLow - dailyHigh2 > atr) | (dailyLow2 - dailyHigh > atr))

    bfvg = (dailyLow > dailyHigh2) & volfilt & atrfilt & locfiltb
    sfvg = (dailyHigh < dailyLow2) & volfilt & atrfilt & locfilts

    is_swing_high = (dailyHigh1 < dailyHigh2) & (high.shift(3) < dailyHigh2) & (high.shift(4) < dailyHigh2)
    is_swing_low = (dailyLow1 > dailyLow2) & (low.shift(3) > dailyLow2) & (low.shift(4) > dailyLow2)

    lastSwingType1_series = pd.Series("none", index=df.index)
    last_swing_high1 = pd.Series(np.nan, index=df.index)
    last_swing_low1 = pd.Series(np.nan, index=df.index)

    for i in range(1, len(df)):
        if is_swing_high.iloc[i]:
            last_swing_high1.iloc[i] = dailyHigh2.iloc[i]
            lastSwingType1_series.iloc[i] = "dailyHigh"
        else:
            last_swing_high1.iloc[i] = last_swing_high1.iloc[i-1]
            lastSwingType1_series.iloc[i] = lastSwingType1_series.iloc[i-1]

        if is_swing_low.iloc[i]:
            last_swing_low1.iloc[i] = dailyLow2.iloc[i]
            lastSwingType1_series.iloc[i] = "dailyLow"
        else:
            last_swing_low1.iloc[i] = last_swing_low1.iloc[i-1]
            lastSwingType1_series.iloc[i] = lastSwingType1_series.iloc[i-1]

    bullishFVG = bfvg & (lastSwingType1_series == "dailyLow")
    bearishFVG = sfvg & (lastSwingType1_series == "dailyHigh")

    bullishFVG_valid = bullishFVG & ~bullishFVG.shift(1).fillna(False)
    bearishFVG_valid = bearishFVG & ~bearishFVG.shift(1).fillna(False)

    bullish_entries = df.index[bullishFVG_valid].tolist()
    bearish_entries = df.index[bearishFVG_valid].tolist()

    for idx in bullish_entries:
        ts = int(df['time'].iloc[idx])
        entries.append({
            'trade_num': trade_num,
            'direction': 'long',
            'entry_ts': ts,
            'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            'entry_price_guess': df['close'].iloc[idx],
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': df['close'].iloc[idx],
            'raw_price_b': df['close'].iloc[idx]
        })
        trade_num += 1

    for idx in bearish_entries:
        ts = int(df['time'].iloc[idx])
        entries.append({
            'trade_num': trade_num,
            'direction': 'short',
            'entry_ts': ts,
            'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            'entry_price_guess': df['close'].iloc[idx],
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': df['close'].iloc[idx],
            'raw_price_b': df['close'].iloc[idx]
        })
        trade_num += 1

    return entries

def _calculate_atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = pd.Series(index=df.index, dtype=float)
    atr.iloc[length-1] = tr.iloc[:length].mean()
    
    for i in range(length, len(df)):
        atr.iloc[i] = (atr.iloc[i-1] * (length - 1) + tr.iloc[i]) / length
    
    return atr