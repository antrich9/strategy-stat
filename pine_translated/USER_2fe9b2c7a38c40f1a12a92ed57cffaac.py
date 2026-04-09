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
    if n < 5:
        return []

    volfilt = volume.shift(1) > volume.rolling(9).mean() * 1.5

    atr_values = np.zeros(n)
    tr = np.zeros(n)
    tr[1:] = np.maximum(high.values[1:] - low.values[1:], 
                        np.maximum(np.abs(high.values[1:] - close.values[:-1]),
                                   np.abs(low.values[1:] - close.values[:-1])))
    alpha = 1.0 / 20.0
    atr_values[0] = tr[0]
    for i in range(1, n):
        atr_values[i] = alpha * tr[i] + (1 - alpha) * atr_values[i-1]
    atr = pd.Series(atr_values, index=df.index)
    atr_filter_vals = atr / 1.5
    atrfilt = ((low - high.shift(2)) > atr_filter_vals) | ((low.shift(2) - high) > atr_filter_vals)

    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    is_up = close > open_
    is_down = close < open_

    is_ob_up = is_down.shift(1) & is_up & (close > high.shift(1))
    is_ob_down = is_up.shift(1) & is_down & (close < low.shift(1))
    is_fvg_up = low > high.shift(2)
    is_fvg_down = high < low.shift(2)

    bfvg = is_fvg_up & volfilt & atrfilt & locfiltb
    sfvg = is_fvg_down & volfilt & atrfilt & locfilts

    ts = pd.to_datetime(df['time'], unit='s', utc=True)
    hour = ts.dt.hour
    minute = ts.dt.minute
    is_morning_window = (hour == 8) & (minute <= 45)
    is_afternoon_window = (hour == 15) & (minute <= 45)
    in_time_window = is_morning_window | is_afternoon_window

    entries = []
    trade_num = 1

    for i in range(2, n):
        if pd.isna(bfvg.iloc[i]) or pd.isna(sfvg.iloc[i]) or pd.isna(in_time_window.iloc[i]):
            continue

        if in_time_window.iloc[i]:
            if bfvg.iloc[i]:
                entry_ts = int(time_col.iloc[i])
                entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time_str,
                    'entry_price_guess': float(close.iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(close.iloc[i]),
                    'raw_price_b': float(close.iloc[i])
                })
                trade_num += 1

            if sfvg.iloc[i]:
                entry_ts = int(time_col.iloc[i])
                entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time_str,
                    'entry_price_guess': float(close.iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(close.iloc[i]),
                    'raw_price_b': float(close.iloc[i])
                })
                trade_num += 1

    return entries