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
    # Ensure DataFrame is sorted by time
    df = df.sort_values('time').reset_index(drop=True)

    # Convert timestamps to datetime (UTC)
    ts = pd.to_datetime(df['time'], unit='s', utc=True)
    df['ts'] = ts
    df['date'] = df['ts'].dt.date

    # ------------------------------------------------------------------
    # Previous day high / low
    # ------------------------------------------------------------------
    daily_agg = df.groupby('date')['high', 'low'].agg({'high': 'max', 'low': 'min'})
    daily_agg.columns = ['dh', 'dl']
    daily_agg = daily_agg.reset_index()
    daily_agg['prev_dh'] = daily_agg['dh'].shift(1)
    daily_agg['prev_dl'] = daily_agg['dl'].shift(1)

    df = df.merge(daily_agg[['date', 'prev_dh', 'prev_dl']], on='date', how='left')

    # ------------------------------------------------------------------
    # Detect sweeps of previous day high/low
    # ------------------------------------------------------------------
    df['sweep_high'] = df['close'] > df['prev_dh']
    df['sweep_low'] = df['close'] < df['prev_dl']

    # Flag per day if any sweep occurred
    df['dh_swept'] = df.groupby('date')['sweep_high'].transform('any')
    df['dl_swept'] = df.groupby('date')['sweep_low'].transform('any')

    # ------------------------------------------------------------------
    # Time‑of‑day filters (morning Long window, afternoon Short window)
    # ------------------------------------------------------------------
    hour = df['ts'].dt.hour
    weekday = df['ts'].dt.dayofweek
    is_weekday = weekday < 5

    df['session_long']  = is_weekday & (hour >= 7) & (hour < 10)
    df['session_short'] = is_weekday & (hour >= 12) & (hour < 15)

    # ------------------------------------------------------------------
    # Order Block (OB) detection
    # ------------------------------------------------------------------
    close = df['close']
    open_s = df['open']
    high = df['high']
    low = df['low']

    is_up   = close > open_s
    is_down = close < open_s

    # Bullish OB (obUp) – pattern 2 bars back, 1 bar back, close > high of 2‑bar‑back
    ob_up = is_down.shift(2) & is_up.shift(1) & (close.shift(1) > high.shift(2))
    ob_up = ob_up.fillna(False)

    # Bearish OB (obDown)
    ob_down = is_up.shift(2) & is_down.shift(1) & (close.shift(1) < low.shift(2))
    ob_down = ob_down.fillna(False)

    # ------------------------------------------------------------------
    # Fair Value Gap (FVG) detection
    # ------------------------------------------------------------------
    fvg_up   = low > high.shift(2)
    fvg_down = high < low.shift(2)
    fvg_up   = fvg_up.fillna(False)
    fvg_down = fvg_down.fillna(False)

    # ------------------------------------------------------------------
    # Stacked OB + FVG conditions
    # ------------------------------------------------------------------
    bull_stack = ob_up & fvg_up
    bear_stack = ob_down & fvg_down

    # ------------------------------------------------------------------
    # Entry conditions
    # ------------------------------------------------------------------
    long_cond  = df['dl_swept'] & bull_stack & df['session_long']
    short_cond = df['dh_swept'] & bear_stack & df['session_short']

    # ------------------------------------------------------------------
    # Build entry list
    # ------------------------------------------------------------------
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if long_cond.iloc[i]:
            ts_int = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entry_time_str = datetime.fromtimestamp(ts_int, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts_int,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            ts_int = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entry_time_str = datetime.fromtimestamp(ts_int, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts_int,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries