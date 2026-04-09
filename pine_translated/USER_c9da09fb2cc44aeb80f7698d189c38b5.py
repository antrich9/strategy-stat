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
    entries = []
    trade_num = 0

    # Timezone offset (1 hour in milliseconds)
    timezone_offset = 3600000

    # Convert timestamps to datetime for hour/minute extraction
    dt_series = pd.to_datetime(df['time'], unit='ms', utc=True)

    # Check DST (simplified: April-October)
    month = dt_series.dt.month
    is_dst = (month >= 4) & (month <= 10)

    # Adjusted time
    adjusted_time_ms = df['time'] + np.where(is_dst, timezone_offset * 2, timezone_offset)
    adjusted_dt = pd.to_datetime(adjusted_time_ms, unit='ms', utc=True)
    adjusted_hour = adjusted_dt.dt.hour
    adjusted_minute = adjusted_dt.dt.minute

    # Trading windows: 07:00-10:59 and 15:00-16:59
    start_hour_1, end_hour_1, end_minute_1 = 7, 10, 59
    start_hour_2, end_hour_2, end_minute_2 = 15, 16, 59

    in_trading_window_1 = (adjusted_hour >= start_hour_1) & (adjusted_hour <= end_hour_1) & ~((adjusted_hour == end_hour_1) & (adjusted_minute > end_minute_1))
    in_trading_window_2 = (adjusted_hour >= start_hour_2) & (adjusted_hour <= end_hour_2) & ~((adjusted_hour == end_hour_2) & (adjusted_minute > end_minute_2))
    in_trading_window = in_trading_window_1 | in_trading_window_2

    # Trend filter: EMA of close
    ema9 = df['close'].ewm(span=9, adjust=False).mean()

    # ATR filter (Wilder)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/20, min_periods=20, adjust=False).mean()

    # Volume filter
    vol_sma = df['volume'].rolling(9).mean()
    volfilt = df['volume'].shift(1) > vol_sma * 1.5

    # ATR filter
    atrfilt = (df['low'] - df['high'].shift(2) > atr / 1.5) | (df['low'].shift(2) - df['high'] > atr / 1.5)

    # Trend filter
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    # Bullish FVG
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb

    # Bearish FVG
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts

    # Order Block conditions
    is_down = df['close'] < df['open']
    is_up = df['close'] > df['open']

    # Bullish OB: down bar followed by up bar, close above high of down bar
    ob_up = (is_down.shift(1)) & (is_up) & (df['close'] > df['high'].shift(1))

    # Bearish OB: up bar followed by down bar, close below low of up bar
    ob_down = (is_up.shift(1)) & (is_down) & (df['close'] < df['low'].shift(1))

    # Nested OB condition (OB at bar 1)
    ob_up_nested = (is_down.shift(2)) & (is_up.shift(1)) & (df['close'].shift(1) > df['high'].shift(2))
    ob_down_nested = (is_up.shift(2)) & (is_down.shift(1)) & (df['close'].shift(1) < df['low'].shift(2))

    # Stacked conditions: FVG + OB alignment
    stacked_bull = bfvg & ob_up & ob_up_nested
    stacked_bear = sfvg & ob_down & ob_down_nested

    # reqDate filter (time-based FVG validity)
    tf = 1  # Assuming intraday
    req_date = datetime.now(timezone.utc).timestamp() * 1000 - 199999999 * tf
    valid_time = df['time'] >= req_date

    # Generate entries
    for i in range(len(df)):
        if i < 3:
            continue

        if not (in_trading_window.iloc[i] and valid_time.iloc[i]):
            continue

        if stacked_bull.iloc[i]:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
        elif stacked_bear.iloc[i]:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })

    return entries