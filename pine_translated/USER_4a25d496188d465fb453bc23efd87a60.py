import pandas as pd
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

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
    # Default parameters (matching Pine script defaults)
    vol_map_period = 20
    vol_multiplier = 1.5
    require_vol_surge = True
    trend_filter = True
    max_trades_per_day = 2

    # Compute indicators
    close = df['close']
    volume = df['volume']

    # Simple moving averages
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    # Volume MA
    vol_ma = volume.rolling(vol_map_period).mean()

    # Conditions
    uptrend = sma50 > sma200
    downtrend = sma50 < sma200

    # Volume surge
    volume_surge = volume > vol_ma * vol_multiplier
    vol_confirmed = ~require_vol_surge | volume_surge  # boolean

    # Trend valid (if trend filter enabled)
    trend_valid = ~trend_filter | uptrend | downtrend

    # Time filter (disabled by default)
    # Compute hour in Europe/London
    ts_series = pd.to_datetime(df['time'], unit='s', utc=True)
    hour_london = ts_series.dt.tz_convert('Europe/London').dt.hour

    use_time_filter = False
    use_all_hours = False
    use_london = False
    use_nyam = False
    use_nypm = False

    in_london = (hour_london >= 8) & (hour_london < 10)
    in_nyam = (hour_london >= 12) & (hour_london < 16)
    in_nypm = (hour_london >= 20) & (hour_london < 21)

    in_time_window = (~use_time_filter) | use_all_hours | in_london | in_nyam | in_nypm

    # Sweep filter (disabled by default)
    use_sweep_filter = False
    long_sweep_ok = pd.Series(True, index=df.index)
    short_sweep_ok = pd.Series(True, index=df.index)

    # Close vs SMA20
    close_above_sma20 = close > sma20
    close_below_sma20 = close < sma20

    # Prepare state
    entries = []
    trade_num = 1
    trades_today = 0
    traded_this_bar = False
    prev_day = None

    # Loop through bars
    for i in df.index:
        # Reset traded_this_bar at start of each bar (simulate var bool reset)
        traded_this_bar = False

        # Compute current day
        ts = ts_series.iloc[i]
        current_day = ts.date()

        # Reset trades_today on new day
        if prev_day is not None and current_day != prev_day:
            trades_today = 0

        prev_day = current_day

        # Check for NaN in required indicators
        if pd.isna(sma20.iloc[i]) or pd.isna(sma50.iloc[i]) or pd.isna(sma200.iloc[i]) or pd.isna(vol_ma.iloc[i]):
            continue

        # tradeAllowed: not in a position? We don't track positions; use trades_today and traded_this_bar
        trade_allowed = (trades_today < max_trades_per_day) and (not traded_this_bar)

        # Apply filters
        if not in_time_window.iloc[i]:
            continue
        if not long_sweep_ok.iloc[i] or not short_sweep_ok.iloc[i]:
            continue

        # Long entry condition
        long_cond = uptrend.iloc[i] and close_above_sma20.iloc[i] and volume_surge.iloc[i] and trend_valid.iloc[i] and trade_allowed
        # Short entry condition
        short_cond = downtrend.iloc[i] and close_below_sma20.iloc[i] and volume_surge.iloc[i] and trend_valid.iloc[i] and trade_allowed

        if long_cond:
            entry_price = close.iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
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
            trade_num += 1
            trades_today += 1
            traded_this_bar = True
        elif short_cond:
            entry_price = close.iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
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
            trade_num += 1
            trades_today += 1
            traded_this_bar = True

    return entries