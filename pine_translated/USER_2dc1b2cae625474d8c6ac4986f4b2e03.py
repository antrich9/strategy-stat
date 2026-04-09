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
    length = 20
    mult = 2.0
    macd_fast_length = 12
    macd_slow_length = 26
    macd_signal_length = 9

    close = df['close']

    basis = close.rolling(length).mean()
    std = close.rolling(length).std()
    upper_band = basis + mult * std
    lower_band = basis - mult * std

    ema_fast = close.ewm(span=macd_fast_length, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow_length, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal_length, adjust=False).mean()

    macd_filter = macd_line > signal_line

    close_above_lower = close > lower_band
    close_above_lower_prev = close.shift(1) <= lower_band.shift(1)
    crossover_lower = close_above_lower & close_above_lower_prev

    close_below_upper = close < upper_band
    close_below_upper_prev = close.shift(1) >= upper_band.shift(1)
    crossunder_upper = close_below_upper & close_below_upper_prev

    long_condition = crossover_lower & macd_filter
    short_condition = crossunder_upper & (~macd_filter)

    entries = []
    trade_num = 1

    for i in range(1, len(df)):
        if pd.isna(basis.iloc[i]) or pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
            continue
        if pd.isna(lower_band.iloc[i-1]) or pd.isna(upper_band.iloc[i-1]):
            continue
        if pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i]):
            continue

        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = float(df['close'].iloc[i])

        if long_condition.iloc[i]:
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
        elif short_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
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