import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    entries = []
    trade_num = 1

    close = df['close']
    high = df['high']
    low = df['low']
    open_prices = df['open']
    ts = df['time']

    # isUp(index) => close[index] > open[index]
    # In Pine: isUp(0) = close > open, isUp(1) = close[1] > open[1], isUp(2) = close[2] > open[2]
    isUp_0 = close > open_prices
    isUp_1 = close.shift(1) > open_prices.shift(1)
    isUp_2 = close.shift(2) > open_prices.shift(2)

    # isDown(index) => close[index] < open[index]
    isDown_0 = close < open_prices
    isDown_1 = close.shift(1) < open_prices.shift(1)
    isDown_2 = close.shift(2) < open_prices.shift(2)

    # isObUp(1) = isDown(2) and isUp(1) and close[1] > high[2]
    obUp = isDown_2 & isUp_1 & (close.shift(1) > high.shift(2))

    # isObDown(1) = isUp(2) and isDown(1) and close[1] < low[2]
    obDown = isUp_2 & isDown_1 & (close.shift(1) < low.shift(2))

    # isFvgUp(0) = low[0] > high[2]
    fvgUp = low > high.shift(2)

    # isFvgDown(0) = high[0] < low[2]
    fvgDown = high < low.shift(2)

    # Previous day high/low using daily data
    df_temp = df.copy()
    df_temp['day'] = pd.to_datetime(df_temp['time'], unit='s', utc=True).dt.date
    daily_agg = df_temp.groupby('day')[['high', 'low']].first().shift(1)
    daily_agg.columns = ['prevDayHigh', 'prevDayLow']
    df_temp = df_temp.merge(daily_agg, left_on='day', right_index=True, how='left')
    prevDayHigh = df_temp['prevDayHigh']
    prevDayLow = df_temp['prevDayLow']

    # Previous day high/low taken
    previousDayHighTaken = high > prevDayHigh
    previousDayLowTaken = low < prevDayLow

    # Current day high/low (4H)
    df_temp['hour'] = pd.to_datetime(df_temp['time'], unit='s', utc=True).dt.hour
    currentDayHigh = df_temp.groupby('day')['high'].transform('max')
    currentDayLow = df_temp.groupby('day')['low'].transform('min')

    # Flags
    flagpdh = previousDayHighTaken & (currentDayLow > prevDayLow)
    flagpdl = previousDayLowTaken & (currentDayHigh < prevDayHigh)

    # London time windows
    dt_utc = pd.to_datetime(ts, unit='s', tz=timezone.utc)
    hours = dt_utc.dt.hour
    minutes = dt_utc.dt.minute
    time_in_minutes = hours * 60 + minutes

    morning_start = 7 * 60 + 45
    morning_end = 9 * 60 + 45
    afternoon_start = 14 * 60 + 45
    afternoon_end = 16 * 60 + 45

    isWithinMorningWindow = (time_in_minutes >= morning_start) & (time_in_minutes < morning_end)
    isWithinAfternoonWindow = (time_in_minutes >= afternoon_start) & (time_in_minutes < afternoon_end)
    in_trading_window = isWithinMorningWindow | isWithinAfternoonWindow

    # Entry conditions
    long_entry = obUp & fvgUp & flagpdh & in_trading_window
    short_entry = obDown & fvgDown & flagpdl & in_trading_window

    n = len(df)
    for i in range(n):
        entry_ts = int(ts.iloc[i])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        entry_price = float(close.iloc[i])

        if long_entry.iloc[i]:
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
        elif short_entry.iloc[i]:
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