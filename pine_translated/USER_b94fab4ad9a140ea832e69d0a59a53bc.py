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
    # Configuration
    tz_name = "Europe/London"
    tz_offset_map = {
        "Europe/London": 0,
        "America/New_York": -5 * 3600,
        "Asia/Tokyo": 9 * 3600,
    }
    tz_offset = tz_offset_map[tz_name]
    fvg_minimum_size = 0.0

    # Extract series
    time = df['time']
    high = df['high']
    low = df['low']
    close = df['close']

    # Determine if bar is within the 10‑11 AM session (target timezone)
    hour = ((time + tz_offset) // 3600) % 24
    in_session = hour == 10

    # Shifted values for the three‑bar FVG pattern
    high_1 = high.shift(1)
    high_2 = high.shift(2)
    low_1 = low.shift(1)
    low_2 = low.shift(2)

    # Bullish FVG: high[2] < low and high[1] < low
    bullish_fvg_cond = (high_2 < low) & (high_1 < low)
    bullish_fvg_size = low - high_2

    # Bearish FVG: low[2] > high and low[1] > high
    bearish_fvg_cond = (low_2 > high) & (low_1 > high)
    bearish_fvg_size = low_2 - high

    # State for the current session
    session_fvg_set = False
    fvg_upper = np.nan
    fvg_lower = np.nan
    is_bullish = None
    traded = False

    entries = []
    trade_num = 1

    for i in range(len(df)):
        ins = in_session.iloc[i]

        # Detect start of a new session
        prev_ins = in_session.iloc[i - 1] if i > 0 else False
        if ins and not prev_ins:
            session_fvg_set = False
            fvg_upper = np.nan
            fvg_lower = np.nan
            is_bullish = None
            traded = False

        if not ins:
            session_fvg_set = False
            fvg_upper = np.nan
            fvg_lower = np.nan
            is_bullish = None
            traded = False
            continue

        # Identify first FVG in the session (require at least two prior bars)
        if not session_fvg_set and i >= 2:
            if bullish_fvg_cond.iloc[i] and bullish_fvg_size.iloc[i] >= fvg_minimum_size:
                fvg_upper = high_2.iloc[i]
                fvg_lower = low.iloc[i]
                is_bullish = True
                session_fvg_set = True
            elif bearish_fvg_cond.iloc[i] and bearish_fvg_size.iloc[i] >= fvg_minimum_size:
                fvg_upper = high.iloc[i]
                fvg_lower = low_2.iloc[i]
                is_bullish = False
                session_fvg_set = True

        # Entry on tap
        if session_fvg_set and not traded:
            if is_bullish:
                if low.iloc[i] <= fvg_upper and low.iloc[i] >= fvg_lower:
                    entry_price = close.iloc[i]
                    entry_ts = int(time.iloc[i])
                    entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': entry_ts,
                        'entry_time': entry_time_str,
                        'entry_price_guess': entry_price,
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': entry_price,
                        'raw_price_b': entry_price,
                    })
                    trade_num += 1
                    traded = True
            else:
                if high.iloc[i] >= fvg_lower and high.iloc[i] <= fvg_upper:
                    entry_price = close.iloc[i]
                    entry_ts = int(time.iloc[i])
                    entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': entry_ts,
                        'entry_time': entry_time_str,
                        'entry_price_guess': entry_price,
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': entry_price,
                        'raw_price_b': entry_price,
                    })