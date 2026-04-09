import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    def pivothigh(series, n):
        ahead = series.rolling(n, min_periods=n).max().shift(-n)
        behind = series.rolling(n, min_periods=n).max().shift(1)
        return np.where((series > ahead) & (series > behind), series, np.nan)

    def pivotlow(series, n):
        ahead = series.rolling(n, min_periods=n).min().shift(-n)
        behind = series.rolling(n, min_periods=n).min().shift(1)
        return np.where((series < ahead) & (series < behind), series, np.nan)

    n = 1

    high = df['high']
    low = df['low']
    close = df['close']

    hih = pivothigh(high, n)
    lol = pivotlow(low, n)

    top = pd.Series(np.nan, index=high.index)
    mask_top = ~pd.isna(hih)
    top[mask_top] = high.shift(n)[mask_top]
    top = top.ffill()

    bottom = pd.Series(np.nan, index=low.index)
    mask_bottom = ~pd.isna(lol)
    bottom[mask_bottom] = low.shift(n)[mask_bottom]
    bottom = bottom.ffill()

    isFvgUp = low > high.shift(2)
    isFvgDown = high < low.shift(2)

    long_cond = isFvgUp & (close.shift(1) > top) & (low.shift(1) < top) & (high.shift(2) < top) & (low > top)
    short_cond = isFvgDown & (close.shift(1) < bottom) & (high.shift(1) > bottom) & (low.shift(2) > bottom) & (high < bottom)

    trades = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(top.iloc[i]) or pd.isna(bottom.iloc[i]):
            continue

        if long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            trades.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            trades.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1

    return trades