import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Ensure columns are present
    close = df['close']
    time = df['time']

    # Parameters (default values from script)
    lengthT3 = 5
    factorT3 = 0.7
    highlightMovementsT3 = True
    crossT3 = True
    inverseT3 = False
    useT3 = True

    # T3 calculation
    def gd(src, length, factor):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factor) - ema2 * factor

    t3 = close
    for _ in range(3):
        t3 = gd(t3, lengthT3, factorT3)

    # t3 up signal
    t3_up = t3 > t3.shift(1)

    # t3SignalsLong
    if useT3:
        if highlightMovementsT3:
            t3SignalsLong = t3_up & (close > t3)
        else:
            t3SignalsLong = close > t3
    else:
        t3SignalsLong = pd.Series(True, index=close.index)

    # cross confirmation
    if crossT3:
        t3SignalsLongCross = t3SignalsLong & ~t3SignalsLong.shift(1).fillna(False)
    else:
        t3SignalsLongCross = t3SignalsLong

    # inverse
    if inverseT3:
        t3SignalsLongFinal = ~t3SignalsLongCross
    else:
        t3SignalsLongFinal = t3SignalsLongCross

    # Stiffness calculation
    maLengthStiffness = 100
    stiffLength = 60
    stiffSmooth = 3
    thresholdStiffness = 90

    sma = close.rolling(window=maLengthStiffness).mean()
    stdev = close.rolling(window=maLengthStiffness).std(ddof=1)
    bound = sma - 0.2 * stdev
    sumAbove = (close > bound).astype(int).rolling(window=stiffLength).sum()
    stiffness = (sumAbove * 100 / stiffLength).ewm(span=stiffSmooth, adjust=False).mean()
    condition_stiffness = stiffness > thresholdStiffness

    # Entry condition
    entry_condition = t3SignalsLongFinal.fillna(False) & condition_stiffness.fillna(False)

    # Generate entries
    entries = []
    trade_num = 1
    for i in df.index:
        if entry_condition[i]:
            ts = int(time[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(close[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
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