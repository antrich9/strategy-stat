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
    # Input parameters (default values from Pine Script)
    d_stats = True
    w_stats = False

    # Initialize tracking variables
    d_ch = np.nan  # Daily current high
    d_cl = np.nan  # Daily current low
    d_co = np.nan  # Daily current open
    d_ph = np.nan  # Daily previous high
    d_pl = np.nan  # Daily previous low
    d_bias = 0  # 1=bullish, -1=bearish, 0=no bias
    d_p_up = False

    entries = []
    trade_num = 0

    # Create day change indicator
    df['date'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.date
    df['new_day'] = df['date'] != df['date'].shift(1)

    for i in range(len(df)):
        row = df.iloc[i]
        new_day = row['new_day'] if i > 0 else True

        # Get previous day's high/low from 1 bar ago (shifted)
        prev_day_high = df['high'].shift(1).iloc[i] if i > 0 else np.nan
        prev_day_low = df['low'].shift(1).iloc[i] if i > 0 else np.nan

        # Previous close from 1 bar ago
        prev_close = df['close'].shift(1).iloc[i] if i > 0 else np.nan

        # Store previous values before update
        prev_ch = d_ch
        prev_cl = d_cl
        prev_ph = d_ph
        prev_pl = d_pl

        # Handle day change - update info
        if d_stats and new_day:
            # Handle bias from previous day
            if not np.isnan(prev_close) and not np.isnan(prev_ph) and not np.isnan(prev_pl):
                if prev_close > prev_ph:
                    d_bias = 1
                elif prev_close < prev_pl:
                    d_bias = -1
                elif prev_close < prev_ph and prev_close > prev_pl:
                    if not np.isnan(d_ch) and d_ch > prev_ph:
                        d_bias = -1
                    elif not np.isnan(d_cl) and d_cl > prev_pl:
                        d_bias = -1
                elif prev_close > prev_pl and prev_close < prev_ph:
                    if not np.isnan(d_ch) and d_ch < prev_ph:
                        d_bias = 1
                    elif not np.isnan(d_cl) and d_cl < prev_pl:
                        d_bias = 1

            if not np.isnan(prev_ch):
                d_ph = prev_ch
                d_pl = prev_cl
                d_p_up = prev_close >= d_co if not np.isnan(d_co) else True

            d_ch = row['high']
            d_cl = row['low']
            d_co = row['open']

        # Initialize if NaN
        if np.isnan(d_ch):
            d_ch = row['high']
            d_cl = row['low']
        else:
            d_ch = max(row['high'], d_ch)
            d_cl = min(row['low'], d_cl)

        # Only process signals after we have valid previous day values
        if i > 0 and not np.isnan(prev_day_high) and not np.isnan(prev_day_low):
            # Get the previous high/low that were set at the start of today
            # For entry logic, we need to track when price HITS yesterday's H/L

            # Check for PDH (Previous Day High) hit
            if row['high'] >= prev_day_high:
                trade_num += 1
                entry_price = row['close']
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(row['time']),
                    'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })

            # Check for PDL (Previous Day Low) hit
            if row['low'] <= prev_day_low:
                trade_num += 1
                entry_price = row['close']
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(row['time']),
                    'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })

    return entries