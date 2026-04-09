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
    open_price = df['open']
    high_price = df['high']
    low_price = df['low']
    close_price = df['close']
    volume_data = df['volume']
    time_data = df['time']

    isUp = close_price > open_price
    isDown = close_price < open_price

    isObUp = isDown.shift(1) & isUp & (close_price > high_price.shift(1))
    isObDown = isUp.shift(1) & isDown & (close_price < low_price.shift(1))

    isFvgUp = low_price > high_price.shift(2)
    isFvgDown = high_price < low_price.shift(2)

    obUp = isObUp.shift(1)
    obDown = isObDown.shift(1)
    fvgUp = isFvgUp
    fvgDown = isFvgDown

    volfilt = volume_data.shift(1) > volume_data.rolling(9).mean() * 1.5
    atr = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / 1.5
    atrfilt = ((low_price - high_price.shift(2) > atr) | (low_price.shift(2) - high_price > atr))
    loc = close_price.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    bfvg = (low_price > high_price.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (high_price < low_price.shift(2)) & volfilt & atrfilt & locfilts

    london_start_morning = pd.to_datetime('07:45:00').time()
    london_end_morning = pd.to_datetime('09:45:00').time()
    london_start_afternoon = pd.to_datetime('14:45:00').time()
    london_end_afternoon = pd.to_datetime('16:45:00').time()

    timestamps = pd.to_datetime(time_data, unit='s', utc=True).tz_convert('Europe/London')
    hours = timestamps.time
    minutes = timestamps

    isWithinMorningWindow = ((hours > london_start_morning) | ((hours == london_start_morning) & (minutes.dt.minute >= 45))) & ((hours < london_end_morning) | ((hours == london_end_morning) & (minutes.dt.minute < 45)))
    isWithinAfternoonWindow = ((hours > london_start_afternoon) | ((hours == london_start_afternoon) & (minutes.dt.minute >= 45))) & ((hours < london_end_afternoon) | ((hours == london_end_afternoon) & (minutes.dt.minute < 45)))
    isWithinTimeWindow = isWithinMorningWindow | isWithinAfternoonWindow

    bullish_condition = (obUp | bfvg) & fvgUp & isWithinTimeWindow
    bearish_condition = (obDown | sfvg) & fvgDown & isWithinTimeWindow

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if i < 2:
            continue
        if pd.isna(obUp.iloc[i]) or pd.isna(bfvg.iloc[i]) or pd.isna(fvgUp.iloc[i]) or pd.isna(obDown.iloc[i]) or pd.isna(sfvg.iloc[i]) or pd.isna(fvgDown.iloc[i]):
            continue
        if bullish_condition.iloc[i]:
            ts = int(time_data.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close_price.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close_price.iloc[i]),
                'raw_price_b': float(close_price.iloc[i])
            })
            trade_num += 1
        elif bearish_condition.iloc[i]:
            ts = int(time_data.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close_price.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close_price.iloc[i]),
                'raw_price_b': float(close_price.iloc[i])
            })
            trade_num += 1

    return entries