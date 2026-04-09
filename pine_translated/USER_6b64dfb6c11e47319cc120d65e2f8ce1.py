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
    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']
    volume = df['volume']
    time_col = df['time']

    n = len(df)

    isUp = close > open_
    isDown = close < open_

    isObUp = pd.Series(False, index=df.index)
    isObDown = pd.Series(False, index=df.index)
    isFvgUp = pd.Series(False, index=df.index)
    isFvgDown = pd.Series(False, index=df.index)

    for i in range(2, n):
        if i + 1 < n:
            isObUp.iloc[i] = isDown.iloc[i + 1] and isUp.iloc[i] and close.iloc[i] > high.iloc[i + 1]
            isObDown.iloc[i] = isUp.iloc[i + 1] and isDown.iloc[i] and close.iloc[i] < low.iloc[i + 1]
        if i - 2 >= 0:
            isFvgUp.iloc[i] = low.iloc[i] > high.iloc[i - 2]
            isFvgDown.iloc[i] = high.iloc[i] < low.iloc[i - 2]

    obUp = isObUp.shift(1).fillna(False)
    obDown = isObDown.shift(1).fillna(False)
    fvgUp = isFvgUp.shift(0).fillna(False)
    fvgDown = isFvgDown.shift(0).fillna(False)

    volfilt = (volume.shift(1) > volume.rolling(9).mean() * 1.5)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()

    loc = close.ewm(span=54, adjust=False).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    bfvg = (low > high.shift(2)) & volfilt & ((low - high.shift(2)) > atr / 1.5) & locfiltb
    sfvg = (high < low.shift(2)) & volfilt & ((low.shift(2) - high) > atr / 1.5) & locfilts

    long_condition = obUp & fvgUp & bfvg
    short_condition = obDown & fvgDown & sfvg

    entries = []
    trade_num = 1

    for i in range(n):
        if i < 3:
            continue

        if pd.isna(loc.iloc[i]) or pd.isna(atr.iloc[i]) or pd.isna(bfvg.iloc[i]):
            continue

        if long_condition.iloc[i]:
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(time_col.iloc[i]),
                'entry_time': datetime.fromtimestamp(time_col.iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

        if short_condition.iloc[i]:
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(time_col.iloc[i]),
                'entry_time': datetime.fromtimestamp(time_col.iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

    return entries