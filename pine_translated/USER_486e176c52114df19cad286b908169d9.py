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
    trade_num = 0

    # Helper functions
    def is_up(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]

    def is_down(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]

    def is_ob_up(idx):
        # Bullish OB: previous candle down, current up, current close > previous high
        if idx < 1:
            return False
        return is_down(idx - 1) and is_up(idx) and df['close'].iloc[idx] > df['high'].iloc[idx - 1]

    def is_ob_down(idx):
        # Bearish OB: previous candle up, current down, current close < previous low
        if idx < 1:
            return False
        return is_up(idx - 1) and is_down(idx) and df['close'].iloc[idx] < df['low'].iloc[idx - 1]

    def is_fvg_up(idx):
        # Bullish FVG: current low > high 2 bars ago
        if idx < 2:
            return False
        return df['low'].iloc[idx] > df['high'].iloc[idx - 2]

    def is_fvg_down(idx):
        # Bearish FVG: current high < low 2 bars ago
        if idx < 2:
            return False
        return df['high'].iloc[idx] < df['low'].iloc[idx - 2]

    # Get timezone info from first timestamp for time filtering
    first_ts = df['time'].iloc[0]
    first_dt = datetime.fromtimestamp(first_ts, tz=timezone.utc)
    gmt_offset = first_dt.hour - datetime.fromtimestamp(first_ts, tz=timezone.utc).hour
    if gmt_offset == 0:
        gmt_offset = 0

    # Create daily high/low arrays
    # Convert timestamps to datetime for grouping
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)

    # Identify new days (at midnight UTC)
    df['day'] = df['datetime'].dt.date

    # Calculate previous day high and low for each row
    prev_day_high = np.nan
    prev_day_low = np.nan

    # Track sweep flags
    flagpdh = False  # PDH swept
    flagpdl = False  # PDL swept

    # Track entry waiting states
    waiting_for_entry = False
    waiting_for_short_entry = False

    # Process each bar
    for i in range(1, len(df)):
        current_ts = df['time'].iloc[i]
        current_dt = datetime.fromtimestamp(current_ts, tz=timezone.utc)
        current_day = df['day'].iloc[i]
        prev_day = df['day'].iloc[i - 1]

        # Check for new day
        if current_day != prev_day:
            # New day - calculate previous day's high and low
            day_mask = df['day'] == prev_day
            if day_mask.any():
                prev_day_high = df.loc[day_mask, 'high'].max()
                prev_day_low = df.loc[day_mask, 'low'].min()
            else:
                prev_day_high = np.nan
                prev_day_low = np.nan

            # Reset flags at start of new day
            flagpdh = False
            flagpdl = False
            waiting_for_entry = False
            waiting_for_short_entry = False

        # Skip if no previous day data
        if pd.isna(prev_day_high) or pd.isna(prev_day_low):
            continue

        # Check for sweep of previous day high
        if df['close'].iloc[i] > prev_day_high:
            flagpdh = True
            waiting_for_entry = True

        # Check for sweep of previous day low
        if df['close'].iloc[i] < prev_day_low:
            flagpdl = True
            waiting_for_short_entry = True

        # Skip if insufficient bars for indicators
        if i < 2:
            continue

        # Calculate OB and FVG conditions at bar i-1 (looking back)
        ob_up = is_ob_up(i - 1)
        ob_down = is_ob_down(i - 1)
        fvg_up = is_fvg_up(i)
        fvg_down = is_fvg_down(i)

        # Get current hour for time filtering
        current_hour = current_dt.hour

        # Check if in time window (0700-0959 for longs, 1200-1459 for shorts)
        in_long_time = 7 <= current_hour < 10
        in_short_time = 12 <= current_hour < 15

        # LONG ENTRY CONDITION
        if flagpdh and waiting_for_entry and not waiting_for_short_entry:
            if ob_up and fvg_up and in_long_time:
                trade_num += 1
                entry_ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entry_price = float(df['close'].iloc[i])

                results.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })

                waiting_for_entry = False

        # SHORT ENTRY CONDITION
        if flagpdl and waiting_for_short_entry and not waiting_for_entry:
            if ob_down and fvg_down and in_short_time:
                trade_num += 1
                entry_ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entry_price = float(df['close'].iloc[i])

                results.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })

                waiting_for_short_entry = False

    return results