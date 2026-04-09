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
    trade_num = 0

    n = len(df)
    if n < 5:
        return results

    dt = df['time'].apply(lambda x: datetime.fromtimestamp(x / 1000, tz=timezone.utc))
    hour = dt.dt.hour
    minute = dt.dt.minute

    isWithinTimeWindow = ((hour == 8) & (minute <= 45)) | ((hour == 15) & (minute <= 45))

    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']

    isDown_1 = close.shift(1) < open_.shift(1)
    isUp_0 = close > open_
    isUp_2 = close.shift(2) > open_.shift(2)
    isDown_1_alt = close.shift(1) < open_.shift(1)

    obUp = isDown_1 & isUp_0 & (close > high.shift(1))
    obDown = isUp_2 & isDown_1_alt & (close.shift(1) < low.shift(2))

    fvgUp = low > high.shift(2)
    fvgDown = high < low.shift(2)

    bullCondition = obUp & fvgUp
    bearCondition = obDown & fvgDown

    volfilt = df['volume'] > df['volume'].rolling(9).mean() * 1.5
    close_prev = close.shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    atrfilt = (low - high.shift(2) > atr) | (low.shift(2) - high > atr)
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    bullSignal = bullCondition & volfilt & atrfilt & locfiltb
    bearSignal = bearCondition & volfilt & atrfilt & locfilts

    bullSignal = bullSignal.fillna(False)
    bearSignal = bearSignal.fillna(False)
    isWithinTimeWindow = isWithinTimeWindow.fillna(False)

    for i in range(1, n):
        if bullSignal.iloc[i] and bullSignal.iloc[i-1] == False and isWithinTimeWindow.iloc[i]:
            if np.isnan(loc.iloc[i]):
                continue
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entry_price = close.iloc[i]
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
        if bearSignal.iloc[i] and bearSignal.iloc[i-1] == False and isWithinTimeWindow.iloc[i]:
            if np.isnan(loc.iloc[i]):
                continue
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entry_price = close.iloc[i]
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

    return results