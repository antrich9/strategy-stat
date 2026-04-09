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
    trade_num = 1

    # Extract datetime components
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['dayofweek'] = df['datetime'].dt.dayofweek  # Monday=0, Sunday=6

    # DST check (Northern Hemisphere)
    is_dst = ((df['month'] >= 3) & ((df['month'] == 3) & (df['dayofweek'] == 6) & (df['day'] >= 25) | (df['month'] > 3)) &
              (df['month'] <= 10) & ((df['month'] == 10) & (df['dayofweek'] == 6) & (df['day'] < 25) | (df['month'] < 10)))

    # UTC+1 adjusted time
    adjusted_offset = (3600000 * (2 if is_dst else 1)).astype('timedelta64[ms]')
    adjusted_time = df['datetime'] + adjusted_offset
    df['adj_hour'] = (adjusted_time.dt.hour % 24).astype(int)
    df['adj_minute'] = adjusted_time.dt.minute.astype(int)

    # Trading windows: 07:00-10:59 and 15:00-16:59
    start_hour_1, end_hour_1, end_minute_1 = 7, 10, 59
    start_hour_2, end_hour_2, end_minute_2 = 15, 16, 59

    in_trading_window_1 = ((df['adj_hour'] > start_hour_1) & (df['adj_hour'] < end_hour_1)) | \
                          ((df['adj_hour'] == start_hour_1) & (df['adj_minute'] >= 0)) | \
                          ((df['adj_hour'] == end_hour_1) & (df['adj_minute'] <= end_minute_1))
    in_trading_window_2 = ((df['adj_hour'] > start_hour_2) & (df['adj_hour'] < end_hour_2)) | \
                          ((df['adj_hour'] == start_hour_2) & (df['adj_minute'] >= 0)) | \
                          ((df['adj_hour'] == end_hour_2) & (df['adj_minute'] <= end_minute_2))
    in_trading_window = in_trading_window_1 | in_trading_window_2

    # New day detection
    df['date'] = df['datetime'].dt.date
    df['is_new_day'] = df['date'].diff() != pd.Timedelta(0)

    # Previous day high/low using daily resampling
    prev_day_high = df['high'].rolling(window=2, min_periods=1).apply(lambda x: x.iloc[:-1].max() if len(x) > 1 else np.nan, raw=True)
    prev_day_low = df['low'].rolling(window=2, min_periods=1).apply(lambda x: x.iloc[:-1].min() if len(x) > 1 else np.nan, raw=True)

    # More accurate: use shift to get previous day values
    daily_high = df['high'].resample('D').max().ffill()
    daily_low = df['low'].resample('D').min().ffill()
    prev_day_high = daily_high.shift(1).reindex(df.index).ffill()
    prev_day_low = daily_low.shift(1).reindex(df.index).ffill()

    # Flags for swept high/low
    flagpdh = (df['close'] > prev_day_high).shift(1).fillna(False).astype(bool)
    flagpdl = (df['close'] < prev_day_low).shift(1).fillna(False).astype(bool)

    # Reset flags on new day
    flagpdh = flagpdh.where(~df['is_new_day'], False)
    flagpdl = flagpdl.where(~df['is_new_day'], False)

    # Fair Value Gap detection (bullish and bearish)
    df['bull_fvg'] = (df['high'] < df['low'].shift(2)) & ((df['low'].shift(1) > df['high'].shift(2)))
    df['bear_fvg'] = (df['low'] > df['high'].shift(2)) & ((df['high'].shift(1) < df['low'].shift(2)))

    # Long entry conditions
    long_cond = (in_trading_window & flagpdl & df['bull_fvg'])
    # Short entry conditions
    short_cond = (in_trading_window & flagpdh & df['bear_fvg'])

    # Generate entries
    in_long_position = False
    in_short_position = False

    for i in range(len(df)):
        if i < 2:
            continue

        entry_price = df['close'].iloc[i]
        entry_ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()

        if long_cond.iloc[i] and not in_long_position and not in_short_position:
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
            in_long_position = True

        elif short_cond.iloc[i] and not in_short_position and not in_long_position:
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
            in_short_position = True

    return entries