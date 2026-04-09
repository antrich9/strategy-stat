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
    results = []
    trade_num = 1

    if len(df) < 3:
        return results

    time_arr = df['time'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    close_arr = df['close'].values
    open_arr = df['open'].values

    dt_times = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in time_arr]

    n = len(df)

    asianHigh = np.nan
    asianLow = np.nan
    tempAsianHigh = np.nan
    tempAsianLow = np.nan
    wasInSession = False
    asianSweptHigh = False
    asianSweptLow = False
    asianSessionEnded = False

    pdHigh = np.nan
    pdLow = np.nan
    tempPdHigh = np.nan
    tempPdLow = np.nan
    pdSweptHigh = False
    pdSweptLow = False

    prev_day = None

    for i in range(n):
        ts = time_arr[i]
        dt = dt_times[i]
        ny_hour = dt.hour

        inAsianSession = ny_hour >= 19

        sessionJustStarted = inAsianSession and not wasInSession

        if sessionJustStarted:
            tempAsianHigh = high_arr[i]
            tempAsianLow = low_arr[i]
        elif inAsianSession:
            if not np.isnan(tempAsianHigh):
                tempAsianHigh = max(tempAsianHigh, high_arr[i])
            else:
                tempAsianHigh = high_arr[i]
            if not np.isnan(tempAsianLow):
                tempAsianLow = min(tempAsianLow, low_arr[i])
            else:
                tempAsianLow = low_arr[i]

        if wasInSession and not inAsianSession:
            asianSessionEnded = True
            asianHigh = tempAsianHigh
            asianLow = tempAsianLow
        else:
            asianSessionEnded = False

        wasInSession = inAsianSession

        current_day = dt.date()

        if prev_day is not None and current_day != prev_day:
            pdHigh = tempPdHigh
            pdLow = tempPdLow
            tempPdHigh = high_arr[i]
            tempPdLow = low_arr[i]
            pdSweptHigh = False
            pdSweptLow = False
        else:
            if np.isnan(tempPdHigh):
                tempPdHigh = high_arr[i]
            else:
                tempPdHigh = max(tempPdHigh, high_arr[i])
            if np.isnan(tempPdLow):
                tempPdLow = low_arr[i]
            else:
                tempPdLow = min(tempPdLow, low_arr[i])

        prev_day = current_day

        if not asianSweptHigh and not np.isnan(asianHigh) and high_arr[i] > asianHigh:
            asianSweptHigh = True

        if not asianSweptLow and not np.isnan(asianLow) and low_arr[i] < asianLow:
            asianSweptLow = True

        if not pdSweptHigh and not np.isnan(pdHigh) and high_arr[i] > pdHigh:
            pdSweptHigh = True

        if not pdSweptLow and not np.isnan(pdLow) and low_arr[i] < pdLow:
            pdSweptLow = True

        if i >= 2:
            c1_idx = i - 2
            c2_idx = i - 1

            c1High = high_arr[c1_idx]
            c1Low = low_arr[c1_idx]
            c2Close = close_arr[c2_idx]
            c2High = high_arr[c2_idx]
            c2Low = low_arr[c2_idx]

            bullishBias = (c2Close > c1High) or (c2Low < c1Low and c2Close > c1Low)
            bearishBias = (c2Close < c1Low) or (c2High > c1High and c2Close < c1High)
        else:
            bullishBias = False
            bearishBias = False

        asianLongOk = asianSweptLow and not asianSweptHigh
        asianShortOk = asianSweptHigh and not asianSweptLow
        pdLongOk = pdSweptLow and not pdSweptHigh
        pdShortOk = pdSweptHigh and not pdSweptLow
        biasLongOk = bullishBias
        biasShortOk = bearishBias

        longSweepOk = asianLongOk and pdLongOk and biasLongOk
        shortSweepOk = asianShortOk and pdShortOk and biasShortOk

        if longSweepOk:
            entry_ts = int(time_arr[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close_arr[i])

            results.append({
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

        if shortSweepOk:
            entry_ts = int(time_arr[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close_arr[i])

            results.append({
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

    return results