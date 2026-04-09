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
    inp1 = False  # Volume Filter
    inp2 = False  # ATR Filter
    inp3 = False  # Trend Filter

    close = df['close']
    high = df['high']
    low = df['low']
    open_col = df['open']
    volume = df['volume']

    # volfilt = inp1 ? volume[1] > ta.sma(volume, 9)*1.5 : true
    vol_sma = volume.rolling(9).mean()
    volfilt = (inp1 == False) | ((volume.shift(1) > vol_sma * 1.5))
    volfilt = volfilt.fillna(True)

    # atr = ta.atr(20) / 1.5
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean() / 1.5

    # atrfilt = inp2 ? ((low - high[2] > atr) or (low[2] - high > atr)) : true
    atrfilt = (inp2 == False) | ((low - high.shift(2) > atr) | (low.shift(2) - high > atr))
    atrfilt = atrfilt.fillna(True)

    # loc = ta.sma(close, 54)
    loc = close.rolling(54).mean()

    # loc2 = loc > loc[1]
    loc2 = loc > loc.shift(1)

    # locfiltb = inp3 ? loc2 : true
    locfiltb = (~inp3) | loc2
    locfiltb = locfiltb.fillna(True)

    # locfilts = inp3 ? not loc2 : true
    locfilts = (~inp3) | (~loc2)
    locfilts = locfilts.fillna(True)

    # bfvg = low > high[2] and volfilt and atrfilt and locfiltb
    bfvg = (low > high.shift(2)) & volfilt & atrfilt & locfiltb
    bfvg = bfvg.fillna(False)

    # sfvg = high < low[2] and volfilt and atrfilt and locfilts
    sfvg = (high < low.shift(2)) & volfilt & atrfilt & locfilts
    sfvg = sfvg.fillna(False)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if bfvg.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = float(close.iloc[i])
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
        elif sfvg.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = float(close.iloc[i])
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