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
    trade_num = 1

    # Helper functions
    def is_up(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]

    def is_down(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]

    def is_ob_up(idx):
        return (is_down(idx + 1) and is_up(idx) and 
                df['close'].iloc[idx] > df['high'].iloc[idx + 1])

    def is_ob_down(idx):
        return (is_up(idx + 1) and is_down(idx) and 
                df['close'].iloc[idx] < df['low'].iloc[idx + 1])

    def is_fvg_up(idx):
        return df['low'].iloc[idx] > df['high'].iloc[idx + 2]

    def is_fvg_down(idx):
        return df['high'].iloc[idx] < df['low'].iloc[idx + 2]

    # Indicators
    vol_sma = df['volume'].rolling(9).mean()
    volfilt = vol_sma * 1.5 < df['volume'].shift(1)

    atr = df['high'] - df['low']
    atr_smoothed = atr.ewm(span=19, adjust=False).mean()
    atr_final = atr_smoothed / 1.5
    atrfilt = ((df['low'] - df['high'].shift(2) > atr_final) | 
               (df['low'].shift(2) - df['high'] > atr_final))

    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    bfvg = ((df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb)
    sfvg = ((df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts)

    # Time windows (London time)
    times = pd.to_datetime(df['time'], unit='ms', utc=True)
    london_times = times.dt.tz_convert('Europe/London')
    hours = london_times.dt.hour
    minutes = london_times.dt.minute

    morning_start = (hours == 8) & (minutes >= 0)
    morning_end = (hours == 9) & (minutes <= 45)
    morning_window = morning_start & morning_end

    afternoon_start = (hours == 15) & (minutes >= 0)
    afternoon_end = (hours == 16) & (minutes <= 45)
    afternoon_window = afternoon_start & afternoon_end

    is_within_time_window = morning_window | afternoon_window

    # Previous day high/low (using 1-day shifted values)
    prev_day_high = df['high'].shift(1).rolling('1D').max().shift(1)
    prev_day_low = df['low'].shift(1).rolling('1D').min().shift(1)

    # OB and FVG conditions
    ob_up = df['close'].shift(2) < df['open'].shift(2) & (df['close'] > df['open']) & (df['close'] > df['high'].shift(2))
    ob_down = df['close'].shift(2) > df['open'].shift(2) & (df['close'] < df['open']) & (df['close'] < df['low'].shift(2))
    fvg_up = is_fvg_up(0)
    fvg_down = is_fvg_down(0)

    # Entry conditions
    long_entry = is_within_time_window & ob_up & fvg_up
    short_entry = is_within_time_window & ob_down & fvg_down

    for i in range(len(df)):
        if i < 3:
            continue
        if pd.isna(df['close'].iloc[i]):
            continue
        if long_entry.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return results