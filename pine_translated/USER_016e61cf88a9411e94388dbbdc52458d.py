import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['ts'] = df['time']
    if df['ts'].iloc[0] > 1e12:
        df['ts'] = df['ts'] / 1000
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df['date'] = df['dt'].dt.date
    df['hour'] = df['dt'].dt.hour

    def is_dst_for_year(year):
        march_day = 8 + (7 - datetime(year, 3, 1).weekday()) % 7
        november_day = 1 + (7 - datetime(year, 11, 1).weekday()) % 7
        start_dst = datetime(year, 3, march_day, 2, 0)
        end_dst = datetime(year, 11, november_day, 2, 0)
        return start_dst, end_dst

    def check_dst(ts_dt):
        year = ts_dt.year
        start_dst, end_dst = is_dst_for_year(year)
        utc_time = ts_dt.replace(tzinfo=timezone.utc)
        start_dst_utc = start_dst.replace(tzinfo=timezone.utc)
        end_dst_utc = end_dst.replace(tzinfo=timezone.utc)
        return start_dst_utc <= utc_time < end_dst_utc

    df['is_dst'] = df['dt'].apply(check_dst)
    df['offset'] = df['is_dst'].apply(lambda x: -1 if x else 0)
    df['hour_est'] = df['hour'] - df['offset']
    df['in_window'] = (df['hour_est'] >= 8) & (df['hour_est'] < 11)

    daily_agg = df.groupby('date')['high'].max().reset_index()
    daily_agg.columns = ['date', 'daily_high']
    daily_low = df.groupby('date')['low'].min().reset_index()
    daily_low.columns = ['date', 'daily_low']
    daily_agg = daily_agg.merge(daily_low, on='date')
    daily_agg['prev_day_high'] = daily_agg['daily_high'].shift(1)
    daily_agg['prev_day_low'] = daily_agg['daily_low'].shift(1)
    df = df.merge(daily_agg[['date', 'prev_day_high', 'prev_day_low']], on='date', how='left')
    df['prev_day_high'] = df['prev_day_high'].ffill()
    df['prev_day_low'] = df['prev_day_low'].ffill()

    df['pdh_breached'] = df['high'] > df['prev_day_high']
    df['pdl_breached'] = df['low'] < df['prev_day_low']

    body_size_pct_c1 = 70
    body_size_pct_c3 = 70
    use_body_c1 = True
    use_body_c3 = True

    df['body_3'] = np.abs(df['close'] - df['open'])
    df['range_3'] = df['high'] - df['low']
    df['body_req_3'] = df['body_3'] >= (df['range_3'] * body_size_pct_c3 / 100.0)

    df['body_1'] = np.abs(df['close'].shift(2) - df['open'].shift(2))
    df['range_1'] = df['high'].shift(2) - df['low'].shift(2)
    df['body_req_1'] = df['body_1'] >= (df['range_1'] * body_size_pct_c1 / 100.0)

    df['bull_cond'] = (
        ((df['low'].shift(2) < df['prev_day_low']) | (df['low'].shift(1) < df['prev_day_low'])) &
        (df['low'].shift(1) < df['low'].shift(2)) &
        (df['high'].shift(1) < df['high'].shift(2)) &
        (df['low'] > df['low'].shift(1)) &
        (df['close'] > df['high'].shift(1))
    )

    df['bear_cond'] = (
        ((df['high'].shift(2) > df['prev_day_high']) | (df['high'].shift(1) > df['prev_day_high'])) &
        (df['high'].shift(1) > df['high'].shift(2)) &
        (df['low'].shift(1) > df['low'].shift(2)) &
        (df['high'] < df['high'].shift(1)) &
        (df['close'] < df['low'].shift(1))
    )

    cond_body_3 = (~use_body_c3) | df['body_req_3']
    cond_body_1 = (~use_body_c1) | df['body_req_1']

    df['valid_bullish'] = df['bull_cond'] & cond_body_3 & cond_body_1
    df['valid_bearish'] = df['bear_cond'] & cond_body_3 & cond_body_1
    df['pattern'] = 0
    df.loc[df['valid_bullish'], 'pattern'] = 1
    df.loc[df['valid_bearish'], 'pattern'] = -1

    entries = []
    trade_num = 1

    for i in range(len(df)):
        row = df.iloc[i]
        if pd.isna(row['prev_day_high']) or pd.isna(row['prev_day_low']):
            continue
        if pd.isna(row['low']) or pd.isna(row['high']):
            continue
        if i < 2:
            continue
        if np.isnan(row['body_3']) or np.isnan(row['range_3']):
            continue
        if np.isnan(row['body_1']) or np.isnan(row['range_1']):
            continue
        if row['pattern'] == 1:
            direction = 'long'
            entry_price = row['close']
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': int(row['ts']),
                'entry_time': row['dt'].isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries