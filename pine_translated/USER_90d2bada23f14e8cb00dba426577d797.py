import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['datetime'].dt.date

    daily_high = df.groupby('date')['high'].max()
    daily_low = df.groupby('date')['low'].min()

    prev_day_high = daily_high.reindex(daily_high.index + pd.Timedelta(days=1))
    prev_day_low = daily_low.reindex(daily_low.index + pd.Timedelta(days=1))

    df['prev_day_high'] = df['date'].map(prev_day_high)
    df['prev_day_low'] = df['date'].map(prev_day_low)

    bullishCondition = (
        ((df['low'].shift(2) < df['prev_day_low']) | (df['low'].shift(1) < df['prev_day_low'])) &
        (df['low'].shift(1) < df['low'].shift(2)) &
        (df['high'].shift(1) < df['high'].shift(2)) &
        (df['low'] > df['low'].shift(1)) &
        (df['close'] > df['high'].shift(1))
    )
    bearishCondition = (
        ((df['high'].shift(2) > df['prev_day_high']) | (df['high'].shift(1) > df['prev_day_high'])) &
        (df['high'].shift(1) > df['high'].shift(2)) &
        (df['low'].shift(1) > df['low'].shift(2)) &
        (df['high'] < df['high'].shift(1)) &
        (df['close'] < df['low'].shift(1))
    )

    bodySizeReqC1 = np.abs(df['close'].shift(2) - df['open'].shift(2)) >= (np.abs(df['high'].shift(2) - df['low'].shift(2)) * 70 / 100)
    bodySizeReqC3 = np.abs(df['close'] - df['open']) >= (np.abs(df['high'] - df['low']) * 70 / 100)

    validBullish = bullishCondition & bodySizeReqC1 & bodySizeReqC3
    validBearish = bearishCondition & bodySizeReqC1 & bodySizeReqC3

    patternBullish = validBullish
    patternBearish = validBearish

    df['datetime_est'] = df['datetime'] - pd.Timedelta(hours=5)
    isInSession = (df['datetime_est'].dt.hour >= 8) & (df['datetime_est'].dt.hour < 11)

    bullishFVG = (df['high'].shift(2) < df['low']) & (df['low'].shift(2) > df['high'].shift(1))
    bearishFVG = (df['low'].shift(2) > df['high']) & (df['high'].shift(2) < df['low'].shift(1))

    pdh_above = df['close'] > df['prev_day_high']
    pdl_below = df['close'] < df['prev_day_low']

    pdhBrokenInSession = pdh_above.cummax()
    pdlBrokenInSession = pdl_below.cummax()

    longConditions = isInSession & pdlBrokenInSession & patternBullish & bullishFVG
    shortConditions = isInSession & pdhBrokenInSession & patternBearish & bearishFVG

    long_signal = longConditions.values
    short_signal = shortConditions.values

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if long_signal[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
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
        elif short_signal[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
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

    return entries