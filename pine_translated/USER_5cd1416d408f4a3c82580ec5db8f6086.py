import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # ensure columns exist
    close = df['close']
    # T3 calculations
    factor = 0.7
    c1 = -factor**3
    c2 = 3*factor**2 + 3*factor**3
    c3 = -6*factor**2 - 3*factor - 3*factor**3
    c4 = 1 + 3*factor + factor**2 + 3*factor**2

    def calc_t3(src, length):
        e1 = src.ewm(span=length, adjust=False).mean()
        e2 = e1.ewm(span=length, adjust=False).mean()
        e3 = e2.ewm(span=length, adjust=False).mean()
        e4 = e3.ewm(span=length, adjust=False).mean()
        e5 = e4.ewm(span=length, adjust=False).mean()
        e6 = e5.ewm(span=length, adjust=False).mean()
        return c1*e6 + c2*e5 + c3*e4 + c4*e3

    t3_25 = calc_t3(close, 25)
    t3_100 = calc_t3(close, 100)
    t3_200 = calc_t3(close, 200)

    long_cond_t3 = (close > t3_25) & (close > t3_100) & (close > t3_200)
    short_cond_t3 = (close < t3_25) & (close < t3_100) & (close < t3_200)

    # TDFI calculation
    price = close * 1000
    mmaLength = 13
    smmaLength = 13
    nLength = 3
    lookback = 13

    mma = price.ewm(span=mmaLength, adjust=False).mean()
    smma = mma.ewm(span=smmaLength, adjust=False).mean()
    impetmma = mma - mma.shift(1)
    impetsmma = smma - smma.shift(1)
    divma = (mma - smma).abs()
    averimpet = (impetmma + impetsmma) / 2.0
    tdf = divma * np.power(averimpet, nLength)
    # highest of abs tdf over lookback * nLength
    window = lookback * nLength
    highest_abs = tdf.abs().rolling(window=window, min_periods=1).max()
    signal = tdf / highest_abs

    filter_high = 0.05
    filter_low = -0.05

    # crossover / crossunder
    cross_above = (signal > filter_high) & (signal.shift(1) <= filter_high)
    cross_below = (signal < filter_low) & (signal.shift(1) >= filter_low)

    # Entry conditions
    long_cond = long_cond_t3 & cross_above
    short_cond = short_cond_t3 & cross_below

    # Transition detection (from false to true)
    long_start = long_cond & ~long_cond.shift(1).fillna(False)
    short_start = short_cond & ~short_cond.shift(1).fillna(False)

    # Build list of entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if long_start.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_start.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries