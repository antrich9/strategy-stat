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
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('time')

    daily_df = df.resample('D').agg({'high': 'last', 'low': 'last', 'close': 'last', 'open': 'last', 'volume': 'sum'})
    daily_df['high_1'] = daily_df['high'].shift(1)
    daily_df['low_1'] = daily_df['low'].shift(1)
    daily_df['high_2'] = daily_df['high'].shift(2)
    daily_df['low_2'] = daily_df['low'].shift(2)
    daily_df['high_3'] = daily_df['high'].shift(3)
    daily_df['low_3'] = daily_df['low'].shift(3)
    daily_df['high_4'] = daily_df['high'].shift(4)
    daily_df['low_4'] = daily_df['low'].shift(4)

    daily_df['is_swing_high'] = (daily_df['high_1'] < daily_df['high_2']) & \
                                 (daily_df['high_3'] < daily_df['high_2']) & \
                                 (daily_df['high_4'] < daily_df['high_2'])
    daily_df['is_swing_low'] = (daily_df['low_1'] > daily_df['low_2']) & \
                                (daily_df['low_3'] > daily_df['low_2']) & \
                                (daily_df['low_4'] > daily_df['low_2'])

    last_swing_type = 'none'
    swing_high_price = np.nan
    swing_low_price = np.nan
    last_swing_type_series = pd.Series('none', index=daily_df.index)

    for i in range(len(daily_df)):
        if daily_df['is_swing_high'].iloc[i]:
            swing_high_price = daily_df['high_2'].iloc[i]
            last_swing_type = 'dailyHigh'
        if daily_df['is_swing_low'].iloc[i]:
            swing_low_price = daily_df['low_2'].iloc[i]
            last_swing_type = 'dailyLow'
        last_swing_type_series.iloc[i] = last_swing_type

    df['swing_type'] = df.index.map(last_swing_type_series)

    df_4h = df.resample('240T').agg({'high': 'last', 'low': 'last', 'close': 'last', 'open': 'last', 'volume': 'sum'})
    df_4h['high_1'] = df_4h['high'].shift(1)
    df_4h['low_1'] = df_4h['low'].shift(1)
    df_4h['high_2'] = df_4h['high'].shift(2)
    df_4h['low_2'] = df_4h['low'].shift(2)
    df_4h['high_3'] = df_4h['high'].shift(3)
    df_4h['low_3'] = df_4h['low'].shift(3)
    df_4h['close_1'] = df_4h['close'].shift(1)

    df_4h['bfvg'] = df_4h['low_1'] > df_4h['high_2']
    df_4h['sfvg'] = df_4h['high_1'] < df_4h['low_2']

    df_4h_expanded = df_4h.reindex(df.index, method='ffill')

    df['bfvg_4h'] = df_4h_expanded['bfvg']
    df['sfvg_4h'] = df_4h_expanded['sfvg']
    df['swing_type'] = df_4h_expanded['swing_type']

    df['is_new_day'] = df.index.to_period('D') != df.index.shift(1, '1D').to_period('D')
    df['is_confirmed'] = True

    df['bullish_entry'] = df['bfvg_4h'] & df['is_confirmed'] & (df['swing_type'] == 'dailyLow')
    df['bearish_entry'] = df['sfvg_4h'] & df['is_confirmed'] & (df['swing_type'] == 'dailyHigh')

    entries = []
    trade_num = 1

    for i in range(1, len(df)):
        if df['bullish_entry'].iloc[i]:
            entry_ts = int(df.index[i].timestamp())
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif df['bearish_entry'].iloc[i]:
            entry_ts = int(df.index[i].timestamp())
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1

    return entries