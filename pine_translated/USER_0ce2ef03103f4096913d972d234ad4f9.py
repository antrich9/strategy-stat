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

    close = df['close']
    open_arr = df['open']
    high = df['high']
    low = df['low']
    time_arr = df['time']

    # London time windows (07:45-09:45 and 15:45-16:45)
    isWithinTimeWindow = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        dt = datetime.fromtimestamp(time_arr.iloc[i], tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        total_min = hour * 60 + minute
        isWithinTimeWindow[i] = ((total_min >= 7 * 60 + 45) and (total_min < 9 * 60 + 45)) or \
                                 ((total_min >= 15 * 60 + 45) and (total_min < 16 * 60 + 45))

    # Previous day high/low using 1-day lookback
    prevDayHigh = np.full(len(df), np.nan)
    prevDayLow = np.full(len(df), np.nan)
    for i in range(1, len(df)):
        dt_current = datetime.fromtimestamp(time_arr.iloc[i], tz=timezone.utc)
        dt_prev = datetime.fromtimestamp(time_arr.iloc[i-1], tz=timezone.utc)
        if dt_current.date() != dt_prev.date():
            prevDayHigh[i] = high.iloc[i-1]
            prevDayLow[i] = low.iloc[i-1]
        else:
            prevDayHigh[i] = prevDayHigh[i-1]
            prevDayLow[i] = prevDayLow[i-1]

    # Current day high/low (240 min)
    currentDayHigh = np.full(len(df), np.nan)
    currentDayLow = np.full(len(df), np.nan)
    for i in range(len(df)):
        dt = datetime.fromtimestamp(time_arr.iloc[i], tz=timezone.utc)
        day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        if dt.hour < 4:
            window_start = day_start.replace(hour=0)
            window_end = day_start.replace(hour=4)
        elif dt.hour < 8:
            window_start = day_start.replace(hour=4)
            window_end = day_start.replace(hour=8)
        elif dt.hour < 12:
            window_start = day_start.replace(hour=8)
            window_end = day_start.replace(hour=12)
        elif dt.hour < 16:
            window_start = day_start.replace(hour=12)
            window_end = day_start.replace(hour=16)
        elif dt.hour < 20:
            window_start = day_start.replace(hour=16)
            window_end = day_start.replace(hour=20)
        else:
            window_start = day_start.replace(hour=20)
            window_end = day_start.replace(hour=23, minute=59, second=59)
        mask = (time_arr >= window_start.timestamp()) & (time_arr < window_end.timestamp())
        if mask.any():
            currentDayHigh[i] = high[mask].max()
            currentDayLow[i] = low[mask].min()

    # OB/FVG conditions
    obUp = np.zeros(len(df), dtype=bool)
    obDown = np.zeros(len(df), dtype=bool)
    fvgUp = np.zeros(len(df), dtype=bool)
    fvgDown = np.zeros(len(df), dtype=bool)

    for i in range(2, len(df)):
        isUp_Current = close.iloc[i] > open_arr.iloc[i]
        isDown_Current = close.iloc[i] < open_arr.iloc[i]
        isUp_Prev1 = close.iloc[i-1] > open_arr.iloc[i-1]
        isDown_Prev1 = close.iloc[i-1] < open_arr.iloc[i-1]
        isUp_Prev2 = close.iloc[i-2] > open_arr.iloc[i-2]
        isDown_Prev2 = close.iloc[i-2] < open_arr.iloc[i-2]

        obUp[i] = isDown_Prev1 and isUp_Current and close.iloc[i] > high.iloc[i-1]
        obDown[i] = isUp_Prev1 and isDown_Current and close.iloc[i] < low.iloc[i-1]
        fvgUp[i] = low.iloc[i] > high.iloc[i-2]
        fvgDown[i] = high.iloc[i] < low.iloc[i-2]

    # Previous day high/low taken
    previousDayHighTaken = np.zeros(len(df), dtype=bool)
    previousDayLowTaken = np.zeros(len(df), dtype=bool)
    for i in range(1, len(df)):
        if not np.isnan(prevDayHigh[i]):
            previousDayHighTaken[i] = high.iloc[i] > prevDayHigh[i]
        if not np.isnan(prevDayLow[i]):
            previousDayLowTaken[i] = low.iloc[i] < prevDayLow[i]

    # Flags
    flagpdh = np.zeros(len(df), dtype=bool)
    flagpdl = np.zeros(len(df), dtype=bool)
    for i in range(1, len(df)):
        if previousDayHighTaken[i] and not np.isnan(currentDayLow[i]) and not np.isnan(prevDayLow[i]) and currentDayLow[i] > prevDayLow[i]:
            flagpdh[i] = True
        elif previousDayLowTaken[i] and not np.isnan(currentDayHigh[i]) and not np.isnan(prevDayHigh[i]) and currentDayHigh[i] < prevDayHigh[i]:
            flagpdl[i] = True

    # Entry conditions
    longCondition = isWithinTimeWindow & obUp & fvgUp & flagpdh
    shortCondition = isWithinTimeWindow & obDown & fvgDown & flagpdl

    for i in range(len(df)):
        if longCondition.iloc[i] if hasattr(longCondition, 'iloc') else longCondition[i]:
            entry_ts = int(time_arr.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
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

        if shortCondition.iloc[i] if hasattr(shortCondition, 'iloc') else shortCondition[i]:
            entry_ts = int(time_arr.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
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