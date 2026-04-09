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
    # Input defaults from Pine Script
    use_setup_a = True
    use_setup_b = False
    use_setup_c = False
    sweep_buffer = 0.0
    disp_atr_min = 0.0
    trade_longs = True
    trade_shorts = True
    one_per_day = True
    atr_len = 14

    # Session definitions (UTC strings)
    asia_session = ("23:00", "05:00")
    london_session = ("07:00", "10:00")
    ny_session = ("14:30", "17:00")

    # Convert timestamps to time strings for session detection
    df = df.copy()
    df['ts'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['time_str'] = df['ts'].dt.strftime("%H:%M")
    df['dayofmonth'] = df['ts'].dt.day

    def in_session(time_str, sess_start, sess_end):
        if sess_start <= sess_end:
            return (time_str >= sess_start) & (time_str < sess_end)
        else:
            return (time_str >= sess_start) | (time_str < sess_end)

    df['in_asia'] = in_session(df['time_str'], *asia_session)
    df['in_london'] = in_session(df['time_str'], *london_session)
    df['in_ny'] = in_session(df['time_str'], *ny_session)

    # Session transitions
    df['asia_start'] = df['in_asia'] & ~df['in_asia'].shift(1).fillna(False)
    df['asia_end'] = ~df['in_asia'] & df['in_asia'].shift(1).fillna(False)

    # Wilder RSI/ATR helper: compute EMA with alpha = 1/len
    def wilder_smooth(series, len_):
        alpha = 1.0 / len_
        return series.ewm(alpha=alpha, adjust=False).mean()

    # True Range
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR(14)
    atr = wilder_smooth(df['tr'], atr_len)

    # Asia range tracking
    asia_high = np.nan
    asia_low = np.nan
    asia_high_lock = np.nan
    asia_low_lock = np.nan
    range_ready = False

    df['asia_high_lock'] = np.nan
    df['asia_low_lock'] = np.nan
    df['range_ready'] = False

    # State variables
    last_trade_day = -1
    swept_hi = False
    swept_lo = False
    bull_armed = False
    bear_armed = False
    bull_setup = ""
    bear_setup = ""
    bull_setup_bar = -1
    bear_setup_bar = -1

    trade_num = 0
    entries = []

    for idx in df.index:
        row = df.loc[idx]

        # Asia range building
        if row['asia_start']:
            asia_high = row['high']
            asia_low = row['low']
            range_ready = False

        if row['in_asia']:
            asia_high = max(asia_high if not np.isnan(asia_high) else row['high'], row['high'])
            asia_low = min(asia_low if not np.isnan(asia_low) else row['low'], row['low'])

        if row['asia_end'] and not np.isnan(asia_high) and not np.isnan(asia_low):
            asia_high_lock = asia_high
            asia_low_lock = asia_low
            range_ready = True

        df.loc[idx, 'asia_high_lock'] = asia_high_lock
        df.loc[idx, 'asia_low_lock'] = asia_low_lock
        df.loc[idx, 'range_ready'] = range_ready

        # New day reset
        new_day = row['dayofmonth'] != last_trade_day if last_trade_day != -1 else True
        if new_day:
            swept_hi = False
            swept_lo = False
            bull_armed = False
            bear_armed = False
            bull_setup = ""
            bear_setup = ""
            bull_setup_bar = -1
            bear_setup_bar = -1

        trade_ok = not one_per_day or (row['dayofmonth'] != last_trade_day)

        # Setup A
        if use_setup_a and range_ready and row['in_london']:
            sweep_pts = atr.iloc[idx] * sweep_buffer
            body = abs(row['close'] - row['open'])
            disp_pts = atr.iloc[idx] * disp_atr_min

            # Bull Setup