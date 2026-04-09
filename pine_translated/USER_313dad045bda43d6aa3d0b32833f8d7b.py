import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Ensure sorted by time
    df = df.sort_values('time').reset_index(drop=True)

    # Compute previous day high, low, close
    # Convert time to datetime (UTC)
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)

    # Get date (in UTC) for grouping
    df['date'] = df['datetime'].dt.date

    # Compute previous day high/low/close
    # For each date, compute the high, low, close of the previous day
    # We can shift the daily high/low/close by one day
    daily = df.groupby('date').agg({'high': 'max', 'low': 'min', 'close': 'last'}).reset_index()
    daily.columns = ['date', 'daily_high', 'daily_low', 'daily_close']

    # Shift daily data to get previous day values
    daily['prev_daily_high'] = daily['daily_high'].shift(1)
    daily['prev_daily_low'] = daily['daily_low'].shift(1)
    daily['prev_daily_close'] = daily['daily_close'].shift(1)

    # Merge back
    df = df.merge(daily[['date', 'prev_daily_high', 'prev_daily_low', 'prev_daily_close']], on='date', how='left')

    # Compute sweep conditions
    df['sweptLow'] = df['low'] < df['prev_daily_low']
    df['sweptHigh'] = df['high'] > df['prev_daily_high']
    df['brokeHigh'] = df['close'] > df['prev_daily_high']
    df['brokeLow'] = df['close'] < df['prev_daily_low']

    # Compute time window condition (in_trading_window)
    # Convert datetime to Europe/London timezone
    df['datetime_london'] = df['datetime'].dt.tz_convert('Europe/London')

    # Extract hour and minute
    df['hour'] = df['datetime_london'].dt.hour
    df['minute'] = df['datetime_london'].dt.minute

    # Define window1: 07:45 to 11:45 (including start, excluding end)
    # Define window2: 14:00 to 14:45
    def in_window(row):
        h = row['hour']
        m = row['minute']
        # Window1: 07:45 <= time < 11:45
        if (h == 7 and m >= 45) or (7 < h < 11) or (h == 11 and m < 45):
            return True
        # Window2: 14:00 <= time < 14:45
        if h == 14 and m >= 0 and m < 45:
            return True
        return False

    df['in_trading_window'] = df.apply(in_window, axis=1)

    # Generate entries
    entries = []
    trade_num = 1

    for i, row in df.iterrows():
        if row['in_trading_window']:
            if row['sweptLow'] and row['brokeHigh']:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(row['time']),
                    'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(row['close']),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(row['close']),
                    'raw_price_b': float(row['close'])
                })
                trade_num += 1
            elif row['sweptHigh'] and row['brokeLow']:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(row['time']),
                    'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(row['close']),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(row['close']),
                    'raw_price_b': float(row['close'])
                })
                trade_num += 1

    return entries