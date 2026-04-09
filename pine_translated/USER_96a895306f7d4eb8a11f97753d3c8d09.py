import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime_utc'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['datetime_est'] = df['datetime_utc'] - timedelta(hours=5)
    df['date_est'] = df['datetime_est'].dt.date
    df['hour_est'] = df['datetime_est'].dt.hour
    df['is_in_session'] = (df['hour_est'] >= 8) & (df['hour_est'] <= 11)
    daily = df.groupby('date_est').agg({'high': 'max', 'low': 'min'}).reset_index()
    daily.columns = ['date_est', 'daily_high', 'daily_low']
    daily['prev_day_high'] = daily['daily_high'].shift(1)
    daily['prev_day_low'] = daily['daily_low'].shift(1)
    df = df.merge(daily[['date_est', 'prev_day_high', 'prev_day_low']], on='date_est', how='left')
    df['open_shifted_1'] = df['open'].shift(1)
    df['high_shifted_1'] = df['high'].shift(1)
    df['low_shifted_1'] = df['low'].shift(1)
    df['close_shifted_1'] = df['close'].shift(1)
    df['open_shifted_2'] = df['open'].shift(2)
    df['high_shifted_2'] = df['high'].shift(2)
    df['low_shifted_2'] = df['low'].shift(2)
    df['close_shifted_2'] = df['close'].shift(2)
    body_size_1 = (df['close_shifted_2'] - df['open_shifted_2']).abs()
    range_1 = df['high_shifted_2'] - df['low_shifted_2']
    body_size_requirement_1 = body_size_1 >= (range_1 * 70 / 100)
    body_size_3 = (df['close'] - df['open']).abs()
    range_3 = df['high'] - df['low']
    body_size_requirement_3 = body_size_3 >= (range_3 * 70 / 100)
    bullishCondition = (
        ((df['low_shifted_2'] < df['prev_day_low']) | (df['low_shifted_1'] < df['prev_day_low'])) &
        (df['low_shifted_1'] < df['low_shifted_2']) &
        (df['high_shifted_1'] < df['high_shifted_2']) &
        (df['low'] > df['low_shifted_1']) &
        (df['close'] > df['high_shifted_1'])
    )
    bearishCondition = (
        ((df['high_shifted_2'] > df['prev_day_high']) | (df['high_shifted_1'] > df['prev_day_high'])) &
        (df['high_shifted_1'] > df['high_shifted_2']) &
        (df['low_shifted_1'] > df['low_shifted_2']) &
        (df['high'] < df['high_shifted_1']) &
        (df['close'] < df['low_shifted_1'])
    )
    validBullish = bullishCondition & body_size_requirement_3 & body_size_requirement_1
    validBearish = bearishCondition & body_size_requirement_3 & body_size_requirement_1
    df['pattern_type'] = np.where(validBullish, 1, np.where(validBearish, -1, 0))
    bullishFVG = (df['high_shifted_2'] < df['low']) & (df['low_shifted_2'] > df['high_shifted_1'])
    bearishFVG = (df['low_shifted_2'] > df['high']) & (df['high_shifted_2'] < df['low_shifted_1'])
    df['fvg_detected'] = bullishFVG | bearishFVG
    df['pdh_cross'] = (df['close_shifted_1'] <= df['prev_day_high']) & (df['close'] > df['prev_day_high'])
    df['pdl_cross'] = (df['close_shifted_1'] >= df['prev_day_low']) & (df['close'] < df['prev_day_low'])
    df['pdh_broken_in_session'] = df.groupby('date_est').apply(lambda x: (x['is_in_session'] & x['pdh_cross']).cummax()).reset_index(level=0, drop=True)
    df['pdl_broken_in_session'] = df.groupby('date_est').apply(lambda x: (x['is_in_session'] & x['pdl_cross']).cummax()).reset_index(level=0, drop=True)
    df['long_condition'] = df['is_in_session'] & df['pdl_broken_in_session'] & (df['pattern_type'] == 1) & df['fvg_detected']
    df['short_condition'] = df['is_in_session'] & df['pdh_broken_in_session'] & (df['pattern_type'] == -1) & df['fvg_detected']
    entries = []
    trade_num = 1
    for i in range(2, len(df)):
        if df['prev_day_high'].isna().iloc[i] or df['prev_day_low'].isna().iloc[i]:
            continue
        if df['long_condition'].iloc[i] or df['short_condition'].iloc[i]:
            direction = 'long' if df['long_condition'].iloc[i] else 'short'
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    return entries