import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Copy to avoid mutating input
    df = df.copy()

    # Convert Unix timestamps to UTC datetime
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)

    # Extract date (without time) for daily grouping
    df['date'] = df['dt'].dt.date

    # Detect start of a new day
    df['isNewDay'] = df['date'] != df['date'].shift(1)
    df['isNewDay'].fillna(False, inplace=True)

    # Identify the bar where hour == 5 and minute == 0 in GMT
    is_14gmt = (df['dt'].dt.hour == 5) & (df['dt'].dt.minute == 0)

    # ---------- Compute openAt14GMT (var float) ----------
    # Simulate Pine's `var float openAt14GMT = na` with forward fill after reset on new day
    openAt14GMT = pd.Series(np.nan, index=df.index, dtype=float)
    for i in range(len(df)):
        if df['isNewDay'].iloc[i]:
            # Reset at start of new day
            openAt14GMT.iloc[i] = np.nan
        elif is_14gmt.iloc[i]:
            # Capture the open price at 14:00 GMT (5:00 UTC)
            openAt14GMT.iloc[i] = df['open'].iloc[i]
        else:
            # Hold previous value
            openAt14GMT.iloc[i] = openAt14GMT.iloc[i - 1] if i > 0 else np.nan

    # ---------- Compute previous day's high and low ----------
    daily_agg = df.groupby('date').agg(
        daily_high=('high', 'max'),
        daily_low=('low', 'min')
    ).reset_index()

    # Shift to get previous day's values
    daily_agg['prev_high'] = daily_agg['daily_high'].shift(1)
    daily_agg['prev_low'] = daily_agg['daily_low'].shift(1)

    # Merge back to main DataFrame
    df = df.merge(daily_agg[['date', 'prev_high', 'prev_low']], on='date', how='left')

    # ---------- Indicator series ----------
    prevMid = (df['prev_high'] + df['prev_low']) / 2.0
    inUpperHalf = openAt14GMT > prevMid
    inLowerHalf = openAt14GMT < prevMid

    # ---------- Generate entries ----------
    entries = []
    trade_num = 1
    trade_taken_today = False

    for i in range(len(df)):
        # Reset daily trade flag at the start of each day
        if df['isNewDay'].iloc[i]:
            trade_taken_today = False

        # Skip bars where required values are missing
        if pd.isna(openAt14GMT.iloc[i]):
            continue
        if pd.isna(df['prev_high'].iloc[i]) or pd.isna(df['prev_low'].iloc[i]):
            continue

        # Only one trade per day
        if trade_taken_today:
            continue

        # Determine direction based on position relative to previous day's midpoint
        if inUpperHalf.iloc[i]:
            direction = 'long'
        elif inLowerHalf.iloc[i]:
            direction = 'short'
        else:
            continue  # equal to midpoint – no entry

        # Prepare entry record
        entry_ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        entry_price = float(df['close'].iloc[i])

        entries.append({
            'trade_num': trade_num,
            'direction': direction,
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
        trade_taken_today = True

    return entries