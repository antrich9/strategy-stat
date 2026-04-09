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
    # Ensure DataFrame is sorted by time
    df = df.sort_values('time').reset_index(drop=True)

    # Optional filters (default False => all filters are inactive)
    inp1 = False  # volume filter
    inp2 = False  # ATR filter
    inp3 = False  # trend filter

    # Volume filter: volume[1] > sma(volume,9)*1.5
    if inp1:
        vol_sma = df['volume'].rolling(9).mean()
        volfilt = df['volume'].shift(1) > vol_sma * 1.5
        volfilt = volfilt.fillna(False)
    else:
        volfilt = pd.Series(True, index=df.index)

    # ATR filter (Wilder ATR 20 periods)
    if inp2:
        tr1 = df['high'] - df['low']
        tr2 = np.abs(df['high'] - df['close'].shift(1))
        tr3 = np.abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=20, adjust=False).mean()
        atrfilt = (df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr)
        atrfilt = atrfilt.fillna(False)
    else:
        atrfilt = pd.Series(True, index=df.index)

    # Trend filter: simple moving average of close 54
    if inp3:
        loc = df['close'].rolling(54).mean()
        loc2 = loc > loc.shift(1)
        locfiltb = loc2
        locfilts = ~loc2
        locfiltb = locfiltb.fillna(False)
        locfilts = locfilts.fillna(False)
    else:
        locfiltb = pd.Series(True, index=df.index)
        locfilts = pd.Series(True, index=df.index)

    # Bullish FVG (bfvg): low > high[2] and filters
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb

    # Bearish FVG (sfvg): high < low[2] and filters
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts

    # Generate entries
    entries = []
    trade_num = 1

    for i in df.index:
        if bfvg.iloc[i]:
            entry_price = df['close'].iloc[i]
            entry_ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
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

        if sfvg.iloc[i]:
            entry_price = df['close'].iloc[i]
            entry_ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
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

    return entries