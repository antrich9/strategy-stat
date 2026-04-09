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

    # London trading windows (7:45-9:45 and 15:45-16:45)
    def is_in_london_window(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        # Morning: 7:45 to 9:45
        morning_start = (hour == 7 and minute >= 45) or (hour >= 8 and hour < 9) or (hour == 9 and minute < 45)
        # Afternoon: 15:45 to 16:45
        afternoon_start = (hour == 15 and minute >= 45) or (hour == 16 and minute < 45)
        return morning_start or afternoon_start

    # Calculate previous day high/low using daily aggregation
    df['day'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.date
    daily_agg = df.groupby('day').agg({'high': 'max', 'low': 'min'}).reset_index()
    daily_agg['day'] = pd.to_datetime(daily_agg['day']).astype(np.int64) // 10**9

    prev_day_high = df['high'].shift(1).where(df['day'] != df['day'].shift(1))
    prev_day_high = prev_day_high.fillna(method='ffill')

    prev_day_low = df['low'].shift(1).where(df['day'] != df['day'].shift(1))
    prev_day_low = prev_day_low.fillna(method='ffill')

    # Previous day high/low taken conditions
    previous_day_high_taken = df['high'] > prev_day_high
    previous_day_low_taken = df['low'] < prev_day_low

    # Current day high/low (rolling 240 period approximation using 10 bars of 240min = 2400min day)
    # Since we don't have 240min data, use current day high/low
    current_day_high = df['high']
    current_day_low = df['low']

    # Flags for liquidity sweeps
    flag_pdh = False
    flag_pdl = False

    for i in range(len(df)):
        if i < 2:
            continue

        # Update flags based on conditions
        if previous_day_high_taken.iloc[i] and current_day_low.iloc[i] > prev_day_low.iloc[i]:
            flag_pdh = True
            flag_pdl = False
        elif previous_day_low_taken.iloc[i] and current_day_high.iloc[i] < prev_day_high.iloc[i]:
            flag_pdl = True
            flag_pdh = False
        else:
            flag_pdl = False
            flag_pdh = False

        # Entry conditions
        long_condition = flag_pdl and previous_day_low_taken.iloc[i]
        short_condition = flag_pdh and previous_day_high_taken.iloc[i]

        ts = int(df['time'].iloc[i])
        if is_in_london_window(ts):
            if long_condition:
                trade_num += 1
                entry_price = float(df['close'].iloc[i])
                entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                results.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': entry_time,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })

            if short_condition:
                trade_num += 1
                entry_price = float(df['close'].iloc[i])
                entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                results.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': entry_time,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })

    return results