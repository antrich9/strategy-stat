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
    # Compute EMAs (Pine ta.ema)
    ema8 = df['close'].ewm(span=8, adjust=False).mean()
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    ema50 = df['close'].ewm(span=50, adjust=False).mean()

    # Detect EMA crossover (bullish) and crossunder (bearish)
    ema8_above_ema20 = ema8 > ema20
    ema8_below_ema20 = ema8 < ema20

    # Crossover: ema8 crosses above ema20
    crossover_long = ema8_above_ema20 & (ema8.shift(1) <= ema20.shift(1))
    # Crossunder: ema8 crosses below ema20
    crossunder_short = ema8_below_ema20 & (ema8.shift(1) >= ema20.shift(1))

    # Additional EMA trend filter
    long_condition = crossover_long & (ema20 > ema50)
    short_condition = crossunder_short & (ema20 < ema50)

    # Time filter: valid trading hours (hour in UTC – adjust timezone if needed)
    hour_series = pd.to_datetime(df['time'], unit='s', utc=True).dt.hour
    is_valid_time = ((hour_series >= 2) & (hour_series < 5)) | ((hour_series >= 10) & (hour_series < 12))

    # Combine entry conditions with time filter
    long_signal = long_condition & is_valid_time
    short_signal = short_condition & is_valid_time

    entries = []
    trade_num = 1

    for i in range(len(df)):
        # Skip bars where any required indicator is NaN
        if pd.isna(ema8.iloc[i]) or pd.isna(ema20.iloc[i]) or pd.isna(ema50.iloc[i]):
            continue

        if long_signal.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
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
        elif short_signal.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
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