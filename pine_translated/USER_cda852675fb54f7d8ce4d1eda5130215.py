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

    # Convert time to datetime for easier manipulation
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['datetime'].dt.date

    # === Daily Swing Detection ===
    # Get daily data
    daily = df.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    daily['prev_high'] = daily['high'].shift(1)
    daily['prev_low'] = daily['low'].shift(1)
    daily['prev2_high'] = daily['high'].shift(2)
    daily['prev2_low'] = daily['low'].shift(2)

    # Swing detection: dailyHigh21 < dailyHigh22 AND dailyHigh11[3] < dailyHigh22 AND dailyHigh11[4] < dailyHigh22
    # In our aggregated daily data: prev_high < prev2_high AND shift(3).high < prev2_high AND shift(4).high < prev2_high
    daily['is_swing_high'] = (
        (daily['prev_high'] < daily['prev2_high']) &
        (daily['high'].shift(3) < daily['prev2_high']) &
        (daily['high'].shift(4) < daily['prev2_high'])
    )
    daily['is_swing_low'] = (
        (daily['prev_low'] > daily['prev2_low']) &
        (daily['low'].shift(3) > daily['prev2_low']) &
        (daily['low'].shift(4) > daily['prev2_low'])
    )

    # Track last swing type per date
    df['last_swing_type'] = 'none'
    for idx, row in daily.iterrows():
        date_val = row['date']
        mask = df['date'] == date_val
        if row['is_swing_high']:
            df.loc[mask, 'last_swing_type'] = 'dailyHigh'
        elif row['is_swing_low']:
            df.loc[mask, 'last_swing_type'] = 'dailyLow'

    # Fill forward last_swing_type
    df['last_swing_type'] = df['last_swing_type'].replace('none', np.nan).ffill().fillna('none')

    # === FVG Detection (simplified using daily extremes) ===
    # Bullish FVG: low > high_4h_2 -> using gap between current low and prior high
    # Bearish FVG: high < low_4h_2 -> using gap between current high and prior low
    daily['bull_fvg'] = daily['low'] > daily['prev2_high']
    daily['bear_fvg'] = daily['high'] < daily['prev2_low']

    # Map FVG back to df
    df['bull_fvg'] = df['date'].map(daily.set_index('date')['bull_fvg']).fillna(False)
    df['bear_fvg'] = df['date'].map(daily.set_index('date')['bear_fvg']).fillna(False)

    # === Trading Windows (London time) ===
    # Note: timezone handling
    def get_trading_window_mask(df):
        london_tz = timezone(timedelta(hours=1))  # BST offset (simplified)
        try:
            dt_local = df['datetime'].dt.tz_convert('Europe/London')
        except:
            dt_local = df['datetime'].dt.tz_convert(timedelta(hours=1))
        
        hour = dt_local.dt.hour
        minute = dt_local.dt.minute
        total_min = hour * 60 + minute
        
        # Window 1: 07:00 - 11:45
        window1_start = 7 * 60
        window1_end = 11 * 60 + 45
        
        # Window 2: 14:00 - 14:45
        window2_start = 14 * 60
        window2_end = 14 * 60 + 45
        
        in_window = (
            ((total_min >= window1_start) & (total_min <= window1_end)) |
            ((total_min >= window2_start) & (total_min <= window2_end))
        )
        return in_window

    df['in_trading_window'] = get_trading_window_mask(df)

    # === Sweep Detection ===
    # Previous day high/low calculation
    df['prev_day_date'] = df['date'].shift(1)
    df['prev_day_high'] = np.nan
    df['prev_day_low'] = np.nan

    # Calculate running high/low within each day
    df['day_high'] = df.groupby('date')['high'].transform('max')
    df['day_low'] = df.groupby('date')['low'].transform('min')

    # Previous day high/low (for sweep detection)
    prev_day_agg = daily[['date', 'high', 'low']].copy()
    prev_day_agg.columns = ['date', 'prev_day_high', 'prev_day_low']
    prev_day_agg['date'] = prev_day_agg['date'].shift(-1)  # Shift to match next day
    prev_day_agg = prev_day_agg.dropna()

    df = df.merge(prev_day_agg, on='date', how='left')
    df['prev_day_high'] = df['prev_day_high'].ffill()
    df['prev_day_low'] = df['prev_day_low'].ffill()

    # Sweep detection
    df['swept_high'] = df['high'] > df['prev_day_high']
    df['swept_low'] = df['low'] < df['prev_day_low']

    # === Build Boolean Series for each condition ===
    bull_fvg_confirmed = df['bull_fvg'] & (df['last_swing_type'] == 'dailyLow')
    bear_fvg_confirmed = df['bear_fvg'] & (df['last_swing_type'] == 'dailyHigh')

    # Combine with trading window and sweep conditions
    bull_entry_cond = bull_fvg_confirmed & df['in_trading_window']
    bear_entry_cond = bear_fvg_confirmed & df['in_trading_window']

    # === Iterate and generate entries ===
    for i in range(len(df)):
        if pd.isna(df['close'].iloc[i]):
            continue

        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = df['close'].iloc[i]

        # Bullish entry
        if bull_entry_cond.iloc[i]:
            entries.append({
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
            trade_num += 1

        # Bearish entry
        if bear_entry_cond.iloc[i]:
            entries.append({
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
            trade_num += 1

    return entries