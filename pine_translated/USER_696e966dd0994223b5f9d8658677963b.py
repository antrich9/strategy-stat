import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['day'] = df['datetime'].dt.date

    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1).fillna(close)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_period = 14
    atr = tr.ewm(alpha=1.0/atr_period, adjust=False).mean()

    d_ph = pd.Series(np.nan, index=df.index)
    d_pl = pd.Series(np.nan, index=df.index)
    day_high = pd.Series(np.nan, index=df.index)
    day_low = pd.Series(np.nan, index=df.index)

    new_day = df['day'] != df['day'].shift(1)

    tracking_started = False

    for i in range(len(df)):
        if new_day.iloc[i]:
            if tracking_started:
                d_ph.iloc[i] = day_high.iloc[i-1]
                d_pl.iloc[i] = day_low.iloc[i-1]
            day_high.iloc[i] = high.iloc[i]
            day_low.iloc[i] = low.iloc[i]
            tracking_started = True
        else:
            day_high.iloc[i] = max(day_high.iloc[i-1], high.iloc[i])
            day_low.iloc[i] = min(day_low.iloc[i-1], low.iloc[i])

    close_prev = close.shift(1)
    short_condition = (close_prev > d_ph)

    entries = []
    trade_num = 0
    current_position = "flat"

    for i in range(1, len(df)):
        if np.isnan(d_ph.iloc[i]) or np.isnan(atr.iloc[i]):
            continue

        if current_position == "flat":
            if short_condition.iloc[i]:
                trade_num += 1
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': df['time'].iloc[i],
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close.iloc[i],
                    'raw_price_b': close.iloc[i]
                })
                current_position = "short"
        elif current_position == "short":
            current_position = "flat"

    return entries