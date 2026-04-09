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
    # Default parameters from strategy
    trend_filter = True
    vol_ma_period = 20
    vol_multiplier = 1.5
    require_vol_surge = True
    max_trades_per_day = 2
    use_time_filter = False
    use_sweep_filter = False
    swing_strength = 5

    n = len(df)
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    time_col = df['time']

    # Calculate SMAs
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    sma_20 = close.rolling(20).mean()
    vol_ma = volume.rolling(vol_ma_period).mean()

    # Calculate ATR (Wilder)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/14, adjust=False).mean()

    # Trend detection
    uptrend = sma_50 > sma_200
    downtrend = sma_50 < sma_200
    trend_valid = not trend_filter or (uptrend | downtrend)

    # Volume surge
    volume_surge = volume > vol_ma * vol_multiplier
    vol_confirmed = not require_vol_surge | volume_surge

    # Initialize state variables
    entries = []
    trade_num = 1
    trades_today = 0
    traded_this_bar = False
    prev_trades_today = 0

    # Track daily changes
    for i in range(1, n):
        # Check for new day (skip first bar)
        if i > 0:
            current_ts = time_col.iloc[i]
            prev_ts = time_col.iloc[i-1]
            current_date = datetime.fromtimestamp(current_ts, tz=timezone.utc).date()
            prev_date = datetime.fromtimestamp(prev_ts, tz=timezone.utc).date()
            if current_date != prev_date:
                trades_today = 0

        # Reset traded_this_bar at start of each bar
        traded_this_bar = False

        # Check if we can trade
        trade_allowed = (trades_today < max_trades_per_day) and not traded_this_bar

        # Entry condition: longTriggered
        cond_uptrend = uptrend.iloc[i] if not pd.isna(uptrend.iloc[i]) else False
        cond_vol = vol_confirmed.iloc[i] if not pd.isna(vol_confirmed.iloc[i]) else False
        cond_trend = trend_valid.iloc[i] if not pd.isna(trend_valid.iloc[i]) else False
        cond_sma20 = close.iloc[i] > sma_20.iloc[i] if not pd.isna(sma_20.iloc[i]) else False
        cond_allowed = trade_allowed

        long_triggered = (cond_uptrend and cond_vol and cond_trend and cond_sma20 and cond_allowed)

        # Generate entry if long triggered
        if long_triggered:
            ts = int(time_col.iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])

            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            }
            entries.append(entry)
            trade_num += 1
            trades_today += 1
            traded_this_bar = True

    return entries