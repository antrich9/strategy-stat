import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Ensure sorted by time
    df = df.sort_values('time').reset_index(drop=True)

    # Convert time to datetime
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)

    # Create date column (date only)
    df['date'] = df['datetime'].dt.date

    # Compute daily high, low, close
    daily = df.groupby('date').agg(
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last')
    ).reset_index()
    daily = daily.sort_values('date').reset_index(drop=True)

    # Compute previous day's close and previous-previous day's high/low
    daily['prev_daily_close'] = daily['close'].shift(1)
    daily['prev_prev_daily_high'] = daily['high'].shift(2)
    daily['prev_prev_daily_low'] = daily['low'].shift(2)

    # Compute bias
    def compute_bias(row):
        close = row['prev_daily_close']
        high = row['prev_prev_daily_high']
        low = row['prev_prev_daily_low']
        if pd.isna(close) or pd.isna(high) or pd.isna(low):
            return None
        if close < low:
            return "bearish"
        elif close > high:
            return "bullish"
        elif close > low and close < high:
            return "ranging"
        else:
            return None

    daily['bias'] = daily.apply(compute_bias, axis=1)

    # Map bias back to df
    bias_map = daily.set_index('date')['bias']
    df['bias'] = df['date'].map(bias_map)

    # Compute FVG signals
    # bullishFVG: high 2 bars ago < low current
    # bearishFVG: low 2 bars ago > high current
    # We'll compute shift(2) for high and low.
    high_shifted2 = df['high'].shift(2)
    low_shifted2 = df['low'].shift(2)

    bullishFVG = (high_shifted2 < df['low']) & (df.index >= 2)
    bearishFVG = (low_shifted2 > df['high']) & (df.index >= 2)

    # Initialize state variables
    ig_active = False
    ig_direction = 0  # 1 for long, -1 for short
    ig_c1_high = np.nan
    ig_c1_low = np.nan
    ig_c3_high = np.nan
    ig_c3_low = np.nan
    ig_validation_end = -1

    # validated resets each bar
    entries = []
    trade_num = 1

    # Iterate over bars
    for i in range(len(df)):
        row = df.iloc[i]

        # Detect FVG at this bar (i)
        bull_fvg = bullishFVG.iloc[i]
        bear_fvg = bearishFVG.iloc[i]

        if bull_fvg:
            ig_active = True
            ig_direction = 1
            ig_c1_high = high_shifted2.iloc[i]  # high of bar i-2
            ig_c1_low = low_shifted2.iloc[i]    # low of bar i-2
            ig_c3_high = row['high']
            ig_c3_low = row['low']
            ig_validation_end = i + 4

        if bear_fvg:
            ig_active = True
            ig_direction = -1
            ig_c1_high = high_shifted2.iloc[i]
            ig_c1_low = low_shifted2.iloc[i]
            ig_c3_high = row['high']
            ig_c3_low = row['low']
            ig_validation_end = i + 4

        # Validate
        validated = False
        if ig_active and i <= ig_validation_end:
            if ig_direction == 1 and row['close'] < ig_c1_high:
                validated = True
            if ig_direction == -1 and row['close'] > ig_c1_low:
                validated = True

        # Entry signals
        system_entry_long = validated and ig_direction == -1
        system_entry_short = validated and ig_direction == 1

        # Bias for this bar
        bias = row['bias']

        # Long entry condition
        if system_entry_long and bias == "bearish":
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
            # Reset ig_active? The script sets ig_active := false after entry in commented code.
            # We'll reset to avoid duplicate entries on same signal.
            ig_active = False

        # Short entry condition (if any)
        # The script has a commented block for short, but not included.
        # However we can include short entry condition as per system_entry_short.
        # It may also require bias == "bullish"? The script's condition is not fully shown.
        # We'll follow the same pattern: if system_entry_short and bias == "bullish":
        if system_entry_short and bias == "bullish":
            entry_ts = int(row['time'])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(row['close'])
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
            ig_active = False

    return entries