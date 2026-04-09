import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # ------------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------------
    df = df.copy()
    # Convert Unix timestamps to UTC datetime
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    # Extract calendar date (UTC) – used for daily aggregation
    df['date'] = df['datetime'].dt.date

    # ------------------------------------------------------------------
    # Compute daily high / low and previous day high / low
    # ------------------------------------------------------------------
    daily = df.groupby('date').agg(
        daily_high=('high', 'max'),
        daily_low=('low', 'min')
    ).reset_index()

    # Shift daily values to obtain previous day high / low
    daily['prev_day_high'] = daily['daily_high'].shift(1)
    daily['prev_day_low'] = daily['daily_low'].shift(1)

    # Merge previous‑day values back onto the intraday bars
    df = df.merge(daily[['date', 'prev_day_high', 'prev_day_low']], on='date', how='left')

    # Drop bars where we don’t have a previous day (first day of dataset)
    df = df.dropna(subset=['prev_day_high', 'prev_day_low'])

    # ------------------------------------------------------------------
    # Helper booleans for daily‑level comparisons
    # ------------------------------------------------------------------
    df['daily_low_gt_prev'] = df['daily_low'] > df['prev_day_low']
    df['daily_high_lt_prev'] = df['daily_high'] < df['prev_day_high']

    # ------------------------------------------------------------------
    # Detect price‑sweep conditions (previous day high / low taken)
    # ------------------------------------------------------------------
    df['prev_high_taken'] = df['high'] > df['prev_day_high']
    df['prev_low_taken'] = df['low'] < df['prev_day_low']

    # Apply Pine Script else‑if logic:
    #   flagpdh = prev_high_taken AND daily_low > prev_day_low
    #   flagpdl = (NOT flagpdh) AND prev_low_taken AND daily_high < prev_day_high
    cond_high = df['prev_high_taken'] & df['daily_low_gt_prev']
    cond_low  = df['prev_low_taken'] & df['daily_high_lt_prev']

    df['flagpdh'] = cond_high
    df['flagpdl'] = (~cond_high) & cond_low

    # ------------------------------------------------------------------
    # London trading‑window detection (07:45‑09:44 and 15:45‑16:44)
    # ------------------------------------------------------------------
    df['dt_london'] = df['datetime'].dt.tz_convert('Europe/London')
    df['hour']   = df['dt_london'].dt.hour
    df['minute'] = df['dt_london'].dt.minute

    # Morning window: 07:45 – 09:44
    morning = (
        ((df['hour'] == 7) & (df['minute'] >= 45)) |
        (df['hour'] == 8) |
        ((df['hour'] == 9) & (df['minute'] <= 44))
    )

    # Afternoon window: 15:45 – 16:44
    afternoon = (
        ((df['hour'] == 15) & (df['minute'] >= 45)) |
        ((df['hour'] == 16) & (df['minute'] <= 44))
    )

    df['in_window'] = morning | afternoon

    # ------------------------------------------------------------------
    # Entry signal – long only (One‑sided ICT)
    # ------------------------------------------------------------------
    df['entry_long'] = df['flagpdl'] & df['in_window']

    # ------------------------------------------------------------------
    # Build entry list
    # ------------------------------------------------------------------
    entries = []
    trade_num = 1
    for _, row in df[df['entry_long']].iterrows():
        entry_ts = int(row['time'])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        entry_price = float(row['close'])

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

    return entries