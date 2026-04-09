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
    # Handle case where time might be datetime or already timestamp
    if df['time'].dtype == 'datetime64[ns]':
        df = df.copy()
        df['time'] = df['time'].astype('int64') // 10**6

    # Ensure required columns exist
    for col in ['time', 'open', 'high', 'low', 'close']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Fill volume if not present
    if 'volume' not in df.columns:
        df['volume'] = 0.0

    # Calculate ATR using Wilder's method
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    period = 14

    # True Range calculation
    tr = np.zeros(len(df) - 1)
    for i in range(1, len(df)):
        hl = high[i] - low[i]
        hcp = abs(high[i] - close[i - 1])
        lcp = abs(low[i] - close[i - 1])
        tr[i - 1] = max(hl, max(hcp, lcp))

    # First ATR is SMA
    first_atr = np.mean(tr[:period])
    atr = np.full(len(df), np.nan)
    atr[period] = first_atr

    # Wilder smoothing
    for i in range(period + 1, len(df)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period

    # Initialize tracking variables
    entries = []
    trade_num = 1
    current_position = "flat"
    entry_price = np.nan
    prev_day_high = np.nan
    prev_day_low = np.nan
    prev_day_close = np.nan
    prev_day_open = np.nan
    ch = np.nan
    cl = np.nan
    co = np.nan
    p_up = False
    prev_long_signal = False
    prev_short_signal = False

    for i in range(1, len(df)):
        high_val = high[i]
        low_val = low[i]
        close_val = close[i]
        open_val = df['open'].values[i]
        current_day = pd.Timestamp(df['time'].iloc[i]).day
        prev_day = pd.Timestamp(df['time'].iloc[i - 1]).day
        new_day = current_day != prev_day

        if new_day:
            if not np.isnan(ch):
                if not np.isnan(prev_day_close) and not np.isnan(co):
                    p_up = prev_day_close >= co
                prev_day_high = ch
                prev_day_low = cl
                prev_day_close = co
                prev_day_open = df['open'].values[i - 1]
            ch = high_val
            cl = low_val
            co = open_val
        else:
            if np.isnan(ch):
                ch = high_val
                cl = low_val
            else:
                ch = max(high_val, ch)
                cl = min(low_val, cl)

        long_signal = False
        short_signal = False

        if not np.isnan(prev_day_high) and not np.isnan(prev_day_low):
            if close_val > prev_day_high and current_position == "flat":
                short_signal = True

            if close_val < prev_day_low and current_position == "flat":
                long_signal = True

            if prev_long_signal and current_position == "flat":
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(close_val),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(close_val),
                    'raw_price_b': float(close_val)
                })
                trade_num += 1
                current_position = "long"
                entry_price = close_val
                prev_long_signal = False

            if prev_short_signal and current_position == "flat":
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(close_val),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(close_val),
                    'raw_price_b': float(close_val)
                })
                trade_num += 1
                current_position = "short"
                entry_price = close_val
                prev_short_signal = False

        prev_long_signal = long_signal
        prev_short_signal = short_signal

    return entries