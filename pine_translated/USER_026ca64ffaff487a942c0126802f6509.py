import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    volMult = 1.5
    atrLen = 14
    maxTradesPerDay = 1

    v = np.where(df['close'] > df['open'], df['volume'], -df['volume'])

    times = pd.to_datetime(df['time'], unit='s', utc=True)
    hours = times.dt.hour
    minutes = times.dt.minute
    seconds = times.dt.second
    hms = hours * 10000 + minutes * 100 + seconds

    data = {}
    raw_avgs = np.full(len(df), np.nan)
    avgs = np.full(len(df), np.nan)

    for i in range(len(df)):
        bucket = hms.iloc[i]
        if bucket not in data:
            data[bucket] = []
        data[bucket].append(v[i])
        arr = np.array(data[bucket])
        raw_avgs[i] = arr.mean()
        avgs[i] = np.abs(arr).mean()

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close.fillna(0), low_close.fillna(0)))
    atr = pd.Series(tr).ewm(alpha=1/atrLen, adjust=False).mean().to_numpy()

    bullBias = raw_avgs > 0
    bearBias = raw_avgs < 0
    bullSurge = v > (avgs * volMult)
    bearSurge = v < -(avgs * volMult)

    dates = times.dt.date

    tradesToday = 0
    lastDate = None
    trade_num = 0
    entries = []

    for i in range(1, len(df)):
        currentDate = dates.iloc[i]

        if currentDate != lastDate:
            tradesToday = 0

        canTrade = tradesToday < maxTradesPerDay

        isFlat = len(entries) == 0 or entries[-1]['exit_ts'] != 0

        longCond = bullBias[i] and bullSurge[i] and canTrade and isFlat and not np.isnan(atr[i])
        shortCond = bearBias[i] and bearSurge[i] and canTrade and isFlat and not np.isnan(atr[i])

        if longCond:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            tradesToday += 1
            lastDate = currentDate
        elif shortCond:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            tradesToday += 1
            lastDate = currentDate

    return entries