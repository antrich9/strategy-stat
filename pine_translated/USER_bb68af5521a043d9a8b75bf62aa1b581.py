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
    # Resample to daily to get previous day's high and low
    df_daily = df.groupby(pd.to_datetime(df['time'], unit='s', utc=True).dt.date).agg({
        'high': 'max',
        'low': 'min'
    })
    prev_day_high = df_daily['high'].shift(1)
    prev_day_low = df_daily['low'].shift(1)
    prev_day_high = prev_day_high.reindex(df.index)
    prev_day_low = prev_day_low.reindex(df.index)
    prev_day_high = prev_day_high.ffill().bfill()
    prev_day_low = prev_day_low.ffill().bfill()

    # ATR calculation
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(df['high'] - df['close'].shift(1).abs(),
                               df['low'] - df['close'].shift(1).abs()))
    atr = tr.ewm(span=20, adjust=False).mean() / 1.5

    # Volume filter
    vol_sma = df['volume'].rolling(9).mean()
    volfilt = df['volume'] > vol_sma * 1.5

    # ATR filter
    atrfilt = (df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr)

    # Trend filter
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    # Bullish FVG: low > high[2]
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb

    # Bearish FVG: high < low[2]
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts

    # Sweep conditions
    bull_sweep = (df['close'] > prev_day_high) & (df['close'].shift(1) <= prev_day_high.shift(1))
    bear_sweep = (df['close'] < prev_day_low) & (df['close'].shift(1) >= prev_day_low.shift(1))

    # Time filter: 07:00-09:59 and 12:00-14:59 in GMT+1
    dt = pd.to_datetime(df['time'], unit='s', utc=True)
    dt_local = dt + pd.Timedelta(hours=1)
    hours = dt_local.dt.hour
    mins = dt_local.dt.minute
    time_mins = hours * 60 + mins
    time_filter = ((time_mins >= 420) & (time_mins < 600)) | ((time_mins >= 720) & (time_mins < 900))

    # Entry conditions: sweep followed by FVG
    long_cond = bull_sweep & bfvg.shift(1) & time_filter
    short_cond = bear_sweep & sfvg.shift(1) & time_filter

    # Iterate and build entries
    trade_num = 1
    entries = []
    start_idx = 55

    for i in range(start_idx, len(df)):
        if long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries