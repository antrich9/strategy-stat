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
    # inp1=False, inp2=False, inp3=False by default (all filters disabled)
    inp1 = False
    inp2 = False
    inp3 = False

    # Indicators
    # SMA for trend (54 periods)
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)

    # Volume SMA (9 periods)
    volume_sma_9 = df['volume'].rolling(9).mean()

    # ATR (20 periods) - Wilder's method
    high_low = df['high'] - df['low']
    high_close_prev = np.abs(df['high'] - df['close'].shift(1))
    low_close_prev = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/20, adjust=False).mean() / 1.5

    # Filters
    volfilt = df['volume'].shift(1) > volume_sma_9.shift(1) * 1.5 if inp1 else pd.Series(True, index=df.index)
    atrfilt = ((df['low'] - df['high'].shift(2)) > atr) | ((df['low'].shift(2) - df['high']) > atr) if inp2 else pd.Series(True, index=df.index)
    locfiltb = loc2 if inp3 else pd.Series(True, index=df.index)
    locfilts = ~loc2 if inp3 else pd.Series(True, index=df.index)

    # FVG conditions
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts

    # Time windows (London time: UTC+0/UTC+1)
    times = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('Europe/London')
    hour = times.dt.hour
    minute = times.dt.minute
    morning_window = ((hour == 7) & (minute >= 45)) | ((hour == 8)) | ((hour == 9) & (minute <= 44))
    afternoon_window = ((hour == 14) & (minute >= 45)) | ((hour == 15)) | ((hour == 16) & (minute <= 44))
    in_time_window = morning_window | afternoon_window

    # Entry conditions
    long_cond = bfvg & in_time_window
    short_cond = sfvg & in_time_window

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(atr.iloc[i]):
            continue
        if long_cond.iloc[i]:
            entry_time = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc)
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': entry_time.isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            entry_time = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc)
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': entry_time.isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries