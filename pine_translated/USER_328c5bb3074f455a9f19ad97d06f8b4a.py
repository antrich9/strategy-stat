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
    if len(df) < 6:
        return []

    df = df.copy().reset_index(drop=True)

    # Swing Detection on 4h/daily equivalent using rolling window
    def detect_swing_high_4h(highs, idx, lookback=5):
        if idx < 4:
            return False
        main_bar = highs.iloc[idx - 2]
        if highs.iloc[idx - 1] < main_bar and highs.iloc[idx - 3] < main_bar and highs.iloc[idx - 4] < main_bar:
            return True
        return False

    def detect_swing_low_4h(lows, idx, lookback=5):
        if idx < 4:
            return False
        main_bar = lows.iloc[idx - 2]
        if lows.iloc[idx - 1] > main_bar and lows.iloc[idx - 3] > main_bar and lows.iloc[idx - 4] > main_bar:
            return True
        return False

    # For daily bias calculation
    df['date'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.date

    # FVG Detection
    df['bullFVG'] = (df.index > 2) & (df['low'] > df['high'].shift(2))
    df['bearFVG'] = (df.index > 2) & (df['high'] < df['low'].shift(2))

    # Detect new trading day
    df['new_day'] = df['date'].diff().fillna(False)

    # Track trend direction with swing detection on local data
    bullish_count = 0
    bearish_count = 0
    trend_direction = "Neutral"
    daily_bias = 0

    df['trend_direction'] = "Neutral"
    df['daily_bias'] = 0
    df['bullishBiasD'] = False
    df['bearishBiasD'] = False

    # Process each bar to maintain state
    for i in range(6, len(df)):
        if df['new_day'].iloc[i]:
            # Get previous day and prior day data for bias calculation
            curr_date = df['date'].iloc[i]
            prev_date_idx = i - 1
            while prev_date_idx >= 0 and df['date'].iloc[prev_date_idx] == curr_date:
                prev_date_idx -= 1
            if prev_date_idx < 0:
                continue

            # Calculate swing highs/lows for previous days
            prev_day_highs = df.loc[df['date'] == df['date'].iloc[prev_date_idx], 'high']
            prev_day_lows = df.loc[df['date'] == df['date'].iloc[prev_date_idx], 'low']
            pHigh = prev_day_highs.max() if len(prev_day_highs) > 0 else df['high'].iloc[prev_date_idx]
            pLow = prev_day_lows.min() if len(prev_day_lows) > 0 else df['low'].iloc[prev_date_idx]

            prev_prev_date_idx = prev_date_idx - 1
            while prev_prev_date_idx >= 0 and df['date'].iloc[prev_prev_date_idx] == df['date'].iloc[prev_prev_date_idx + 1]:
                prev_prev_date_idx -= 1
            if prev_prev_date_idx < 0:
                prev_prev_date_idx = prev_date_idx - 1

            yHigh = df['high'].iloc[prev_date_idx] if prev_date_idx >= 0 else 0
            yLow = df['low'].iloc[prev_date_idx] if prev_date_idx >= 0 else 0

            prev_day2_highs = df.loc[df['date'] == df['date'].iloc[prev_prev_date_idx], 'high'] if prev_prev_date_idx >= 0 else pd.Series([0])
            prev_day2_lows = df.loc[df['date'] == df['date'].iloc[prev_prev_date_idx], 'low'] if prev_prev_date_idx >= 0 else pd.Series([0])
            pHigh = prev_day2_highs.max() if len(prev_day2_highs) > 0 else df['high'].iloc[prev_prev_date_idx]
            pLow = prev_day2_lows.min() if len(prev_day2_lows) > 0 else df['low'].iloc[prev_prev_date_idx]

            # Determine daily bias
            if trend_direction == "Bearish" and yHigh > pHigh:
                daily_bias = 1
            elif trend_direction == "Bullish" and yLow < pLow:
                daily_bias = -1
            else:
                daily_bias = 0

        df.at[df.index[i], 'daily_bias'] = daily_bias
        df.at[df.index[i], 'bullishBiasD'] = (daily_bias == 1)
        df.at[df.index[i], 'bearishBiasD'] = (daily_bias == -1)
        df.at[df.index[i], 'trend_direction'] = trend_direction

        # Update swing counts for trend
        if detect_swing_high_4h(df['high'], i):
            bullish_count += 1
            bearish_count = 0
        if detect_swing_low_4h(df['low'], i):
            bearish_count += 1
            bullish_count = 0

        if bullish_count > 1:
            trend_direction = "Bullish"
        elif bearish_count > 1:
            trend_direction = "Bearish"

    # Pullback detection for CISD-like logic
    df['bearishPullback'] = df['close'].shift(1) > df['open'].shift(1)
    df['bullishPullback'] = df['close'].shift(1) < df['open'].shift(1)

    # Trading windows (simplified - full time-based filtering would need hour extraction)
    df['hour'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.hour
    df['in_trading_window'] = ((df['hour'] >= 7) & (df['hour'] < 10)) | ((df['hour'] >= 13) & (df['hour'] < 15))

    # Entry conditions based on FVG + Bias + Trading window
    df['longCondition'] = df['bullFVG'] & df['bullishBiasD'] & df['in_trading_window']
    df['shortCondition'] = df['bearFVG'] & df['bearishBiasD'] & df['in_trading_window']

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if df['longCondition'].iloc[i]:
            entry_price = df['close'].iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()

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

        if df['shortCondition'].iloc[i]:
            entry_price = df['close'].iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()

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

    return entries