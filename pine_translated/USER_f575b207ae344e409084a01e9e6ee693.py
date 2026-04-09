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
    inp1 = False
    inp2 = False
    inp3 = False

    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']

    def is_up(idx):
        return close.iloc[idx] > open_.iloc[idx]

    def is_down(idx):
        return close.iloc[idx] < open_.iloc[idx]

    def is_ob_up(idx):
        return is_down(idx + 1) and is_up(idx) and close.iloc[idx] > high.iloc[idx + 1]

    def is_ob_down(idx):
        return is_up(idx + 1) and is_down(idx) and close.iloc[idx] < low.iloc[idx + 1]

    def is_fvg_up(idx):
        return low.iloc[idx] > high.iloc[idx + 2]

    def is_fvg_down(idx):
        return high.iloc[idx] < low.iloc[idx + 2]

    ob_up = pd.Series([False] * len(df), index=df.index)
    ob_down = pd.Series([False] * len(df), index=df.index)
    fvg_up = pd.Series([False] * len(df), index=df.index)
    fvg_down = pd.Series([False] * len(df), index=df.index)

    for i in range(2, len(df)):
        ob_up.iloc[i] = is_ob_up(i)
        ob_down.iloc[i] = is_ob_down(i)
        fvg_up.iloc[i] = is_fvg_up(i)
        fvg_down.iloc[i] = is_fvg_down(i)

    vol_filt = pd.Series(True, index=df.index)
    if inp1:
        vol_sma = volume.rolling(9).mean()
        vol_filt = volume.shift(1) > vol_sma * 1.5

    def wilder_atr(data_high, data_low, data_close, length):
        tr = pd.concat([
            data_high - data_low,
            (data_high - data_close.shift(1)).abs(),
            (data_low - data_close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = pd.Series(np.nan, index=data_high.index)
        if len(tr) >= length:
            atr.iloc[length - 1] = tr.iloc[:length].mean()
            multiplier = 2.0 / (length + 1)
            for i in range(length, len(tr)):
                atr.iloc[i] = atr.iloc[i - 1] * (1 - multiplier) + tr.iloc[i] * multiplier
        return atr

    atr = wilder_atr(high, low, close, 20) / 1.5

    atrfilt = pd.Series(True, index=df.index)
    if inp2:
        atrfilt = (low - high.shift(2) > atr) | (low.shift(2) - high > atr)

    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)

    locfiltb = pd.Series(True, index=df.index)
    locfilts = pd.Series(True, index=df.index)
    if inp3:
        locfiltb = loc2
        locfilts = ~loc2

    bfvg = (low > high.shift(2)) & vol_filt & atrfilt & locfiltb
    sfvg = (high < low.shift(2)) & vol_filt & atrfilt & locfilts

    bull_entry = (ob_up & fvg_up) | (bfvg & locfiltb) if inp3 else (ob_up & fvg_up) | bfvg
    bear_entry = (ob_down & fvg_down) | (sfvg & locfilts) if inp3 else (ob_down & fvg_down) | sfvg

    entries = []
    trade_num = 1

    for i in range(2, len(df)):
        if np.isnan(loc.iloc[i]):
            continue

        direction = None
        if bull_entry.iloc[i]:
            direction = 'long'
        elif bear_entry.iloc[i]:
            direction = 'short'

        if direction:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

    return entries