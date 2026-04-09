import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    entries = []
    trade_num = 1

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # Volume filter
    vol_sma = volume.rolling(9).mean()
    volfilt = volume > vol_sma * 1.5

    # ATR filter (Wilder ATR 20)
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(20).mean() / 1.5

    # Trend filter (SMA 54)
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    # Bullish FVG: low > high[2]
    bfvg = low > high.shift(2)
    # Bearish FVG: high < low[2]
    sfvg = high < low.shift(2)

    # London time window (07:45-09:44 and 14:45-16:44)
    def is_within_london_window(ts_ms):
        ts_sec = ts_ms / 1000.0
        dt = datetime.utcfromtimestamp(ts_sec)
        year = dt.year
        month = dt.month
        day = dt.day
        
        # Last Sunday of March for DST start
        march31 = datetime(year, 3, 31)
        dst_start_day = 31 - ((march31.weekday() + 1) % 7)
        dst_start = datetime(year, 3, dst_start_day)
        
        # Last Sunday of October for DST end
        oct31 = datetime(year, 10, 31)
        dst_end_day = 31 - ((oct31.weekday() + 1) % 7)
        dst_end = datetime(year, 10, dst_end_day)
        
        current_dt = datetime(year, month, day)
        is_dst = dst_start <= current_dt < dst_end
        offset_hours = 1 if is_dst else 0
        
        london_dt = dt.replace(hour=(dt.hour - offset_hours) % 24)
        hour = london_dt.hour
        minute = london_dt.minute
        hm = hour * 60 + minute
        
        morning_start = 7 * 60 + 45
        morning_end = 9 * 60 + 44
        afternoon_start = 14 * 60 + 45
        afternoon_end = 16 * 60 + 44
        
        return (morning_start <= hm <= morning_end) or (afternoon_start <= hm <= afternoon_end)

    # Combine all long conditions
    long_cond = bfvg & volfilt & (low - high.shift(2) > atr) & locfiltb
    # Combine all short conditions
    short_cond = sfvg & volfilt & (low.shift(2) - high > atr) & locfilts

    for i in range(len(df)):
        if pd.isna(atr.iloc[i]) or pd.isna(loc.iloc[i]):
            continue
        
        row_ts = df['time'].iloc[i]
        
        # Long entries
        if long_cond.iloc[i] and is_within_london_window(row_ts):
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(row_ts),
                'entry_time': datetime.fromtimestamp(row_ts / 1e3, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        
        # Short entries
        if short_cond.iloc[i] and is_within_london_window(row_ts):
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(row_ts),
                'entry_time': datetime.fromtimestamp(row_ts / 1e3, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1

    return entries