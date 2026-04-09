def generate_entries(df: pd.DataFrame) -> list:
    import pandas as pd
    import numpy as np
    from datetime import datetime, timezone

    df = df.copy()
    # Convert time to datetime
    df['time_dt'] = pd.to_datetime(df['time'], unit='s')
    # Day
    df['day'] = df['time_dt'].dt.date

    # Daily high/low
    daily = df.groupby('day').agg(high=('high', 'max'), low=('low', 'min')).reset_index()
    daily = daily.sort_values('day')
    daily['prev_day_high'] = daily['high'].shift(1)
    daily['prev_day_low'] = daily['low'].shift(1)
    # Merge back
    df = df.merge(daily[['day', 'prev_day_high', 'prev_day_low']], on='day', how='left')

    # New day flag
    df['is_new_day'] = ~df['day'].duplicated(keep='first')
    # Actually better:
    # df['is_new_day'] = df['day'] != df['day'].shift(1)
    # Shift yields previous row day; first row has NaN, treat as True.
    df['is_new_day'] = df['day'].ne(df['day'].shift(1))
    # Set first row as new day
    df.loc[df.index[0], 'is_new_day'] = True

    entries = []
    trade_num = 1
    pos = 0  # 0 flat, 1 long, -1 short

    # iterate
    for i, row in df.iterrows():
        # skip rows where required prev_day levels are NaN
        if pd.isna(row['prev_day_high']) and pd.isna(row['prev_day_low']):
            # cannot have any entry
            continue

        # If we are flat (pos==0) we can enter either side
        if pos == 0:
            # Long entry: price touches prev_day_high
            if row['high'] >= row['prev_day_high']:
                entry_price = row['prev_day_high']
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(row['time']),
                    'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
                pos = 1
                # after entering long, skip short entry this day
                # continue to next bars, pos !=0 ensures no more entries
                # proceed to next iteration
                continue

            # Short entry: price touches prev_day_low
            if row['low'] <= row['prev_day_low']:
                entry_price = row['prev_day_low']
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(row['time']),
                    'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
                pos = -1
                continue

        # If we have a position, we remain in that until we manually close (not needed)
        # In this simulation we stay in position across days (no exit)
        # So pos stays same across subsequent bars.

        # If we want to support multiple entries after position closes (i.e., after exit), we would need exit logic.
        # Not required.

    return entries