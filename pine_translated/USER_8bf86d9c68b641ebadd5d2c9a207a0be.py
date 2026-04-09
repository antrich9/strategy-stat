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
    results = []
    trade_num = 1

    if len(df) < 20:
        return results

    time = df['time'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    # Helper function to convert timestamp to datetime
    def ts_to_iso(ts):
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    # Detect new day changes
    ts_series = pd.Series(time)
    daily_changes = ts_series.diff().fillna(0) != 0
    new_day = daily_changes.values

    # Calculate rolling highs/lows for swing detection (simplified daily swing detection)
    # Using rolling max/min with span to simulate daily swings
    window_3 = 3
    swing_high_daily = pd.Series(high).rolling(window=window_3, min_periods=window_3).max().shift(1)
    swing_low_daily = pd.Series(low).rolling(window=window_3, min_periods=window_3).min().shift(1)

    # Last swing type tracking (var string in Pine)
    last_swing_high = np.nan
    last_swing_low = np.nan
    lastSwingType = "none"

    # Track swept high/low (Previous Day High/Low)
    var_pdHigh = np.nan
    var_pdLow = np.nan
    tempHigh = np.nan
    tempLow = np.nan
    sweptHigh = False
    sweptLow = False

    # Detect swing highs/lows
    is_swing_high = np.zeros(len(df), dtype=bool)
    is_swing_low = np.zeros(len(df), dtype=bool)

    # For swing detection, use higher timeframe approximations
    # Since we don't have daily data, we'll use a simplified approach
    # Looking for local highs/lows over larger windows
    for i in range(5, len(df)):
        # Swing high: current high is higher than previous and next few bars, and higher than swing_high_daily
        if i >= 3:
            is_local_high = True
            for j in range(1, 4):
                if i - j >= 0 and high[i] <= high[i - j]:
                    is_local_high = False
                    break
                if i + j < len(high) and high[i] <= high[i + j]:
                    is_local_high = False
                    break
            if is_local_high and high[i] > swing_high_daily.iloc[i] if not pd.isna(swing_high_daily.iloc[i]) else False:
                is_swing_high[i] = True

        # Swing low: current low is lower than previous and next few bars, and lower than swing_low_daily
        if i >= 3:
            is_local_low = True
            for j in range(1, 4):
                if i - j >= 0 and low[i] >= low[i - j]:
                    is_local_low = False
                    break
                if i + j < len(low) and low[i] >= low[i + j]:
                    is_local_low = False
                    break
            if is_local_low and low[i] < swing_low_daily.iloc[i] if not pd.isna(swing_low_daily.iloc[i]) else False:
                is_swing_low[i] = True

    # Trading windows (London time based on UTC timestamps)
    # First window: 07:45 - 11:45
    # Second window: 14:00 - 14:45
    in_trading_window = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        dt = datetime.fromtimestamp(time[i], tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        total_minutes = hour * 60 + minute
        # Window 1: 465-705 minutes (7:45 - 11:45)
        if 465 <= total_minutes < 705:
            in_trading_window[i] = True
        # Window 2: 840-885 minutes (14:00 - 14:45)
        elif 840 <= total_minutes < 885:
            in_trading_window[i] = True

    # Calculate 4H data approximations from 15min data
    # Group by 4H intervals
    four_hour_high = pd.Series(high).rolling(window=16, min_periods=16).max().shift(1)
    four_hour_low = pd.Series(low).rolling(window=16, min_periods=16).min().shift(1)
    four_hour_close = pd.Series(close).shift(1)

    # FVG detection on 4H approximations
    # Bullish FVG: current 4H low > 4H high 2 periods ago
    # Bearish FVG: current 4H high < 4H low 2 periods ago
    bfvg = np.zeros(len(df), dtype=bool)
    sfvg = np.zeros(len(df), dtype=bool)

    for i in range(2, len(df)):
        if i >= 2:
            current_4h_low = four_hour_low.iloc[i] if not pd.isna(four_hour_low.iloc[i]) else 0
            high_2_periods_ago = four_hour_high.iloc[i-2] if not pd.isna(four_hour_high.iloc[i-2]) else 0
            if current_4h_low > high_2_periods_ago and high_2_periods_ago > 0:
                bfvg[i] = True

            current_4h_high = four_hour_high.iloc[i] if not pd.isna(four_hour_high.iloc[i]) else 0
            low_2_periods_ago = four_hour_low.iloc[i-2] if not pd.isna(four_hour_low.iloc[i-2]) else 0
            if current_4h_high < low_2_periods_ago and low_2_periods_ago > 0:
                sfvg[i] = True

    # Process each bar
    for i in range(len(df)):
        # Update PDH/PDL on new day
        if new_day[i] and i > 0:
            var_pdHigh = tempHigh
            var_pdLow = tempLow
            tempHigh = high[i]
            tempLow = low[i]
            sweptHigh = False
            sweptLow = False
        else:
            if pd.isna(tempHigh) or np.isnan(tempHigh):
                tempHigh = high[i]
            else:
                tempHigh = max(tempHigh, high[i])
            if pd.isna(tempLow) or np.isnan(tempLow):
                tempLow = low[i]
            else:
                tempLow = min(tempLow, low[i])

        # Update swing type
        if is_swing_high[i]:
            last_swing_high = high[i]
            lastSwingType = "dailyHigh"
        if is_swing_low[i]:
            last_swing_low = low[i]
            lastSwingType = "dailyLow"

        # Check for sweep of previous day high/low
        if not pd.isna(var_pdHigh) and not np.isnan(var_pdHigh):
            if not sweptHigh and high[i] > var_pdHigh:
                sweptHigh = True
        if not pd.isna(var_pdLow) and not np.isnan(var_pdLow):
            if not sweptLow and low[i] < var_pdLow:
                sweptLow = True

        # Entry conditions based on Pine Script
        # Long: bfvg_condition and barstate.isconfirmed and lastSwingType11 == "dailyLow"
        # Short: sfvg_condition and barstate.isconfirmed and lastSwingType11 == "dailyHigh"
        # Plus in_trading_window condition

        in_window = in_trading_window[i]
        is_confirmed = True  # All bars in our data are confirmed

        if in_window and is_confirmed:
            # Long entry condition
            if bfvg[i] and lastSwingType == "dailyLow":
                entry_price = close[i]
                results.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(time[i]),
                    'entry_time': ts_to_iso(time[i]),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1

            # Short entry condition
            elif sfvg[i] and lastSwingType == "dailyHigh":
                entry_price = close[i]
                results.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(time[i]),
                    'entry_time': ts_to_iso(time[i]),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1

    return results