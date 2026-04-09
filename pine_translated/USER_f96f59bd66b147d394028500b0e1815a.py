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
    n = len(df)
    if n < 5:
        return []

    # Resample to 4H
    df_4h = df.set_index('time').resample('240min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    df_4h.rename(columns={'time': 'time_4h'}, inplace=True)

    # 4H arrays
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    volume_4h = df_4h['volume'].values

    # 4H shift arrays (avoid lookahead by using shift on resampled data)
    high_4h_1 = np.roll(high_4h, 1)
    low_4h_1 = np.roll(low_4h, 1)
    close_4h_1 = np.roll(close_4h, 1)
    open_4h_1 = np.roll(open_4h, 1)
    high_4h_2 = np.roll(high_4h, 2)
    low_4h_2 = np.roll(low_4h, 2)

    # Daily data via resample
    df_daily = df.set_index('time').resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna().reset_index()
    df_daily.rename(columns={'time': 'time_daily'}, inplace=True)

    dailyHigh = df_daily['high'].values
    dailyLow = df_daily['low'].values

    # Daily shift arrays (no lookahead)
    dailyHigh11 = dailyHigh
    dailyLow11 = dailyLow
    dailyHigh21 = np.roll(dailyHigh, 1)
    dailyLow21 = np.roll(dailyLow, 1)
    dailyHigh22 = np.roll(dailyHigh, 2)
    dailyLow22 = np.roll(dailyLow, 2)

    # Swing detection (5 bars needed: 2 past + current + 2 future)
    valid_swing = np.arange(n) >= 2
    is_swing_high_mask = np.where(
        valid_swing,
        (np.greater(dailyHigh21, dailyHigh22, where=~np.isnan(dailyHigh21) & ~np.isnan(dailyHigh22)) | (np.isnan(dailyHigh21) & np.isnan(dailyHigh22))) &
        (np.less(dailyHigh11[3:], dailyHigh22[3:], where=~np.isnan(dailyHigh11[3:]) & ~np.isnan(dailyHigh22[3:])) | (np.isnan(dailyHigh11[3:]) & np.isnan(dailyHigh22[3:]))),
        False
    )
    is_swing_low_mask = np.where(
        valid_swing,
        (np.less(dailyLow21, dailyLow22, where=~np.isnan(dailyLow21) & ~np.isnan(dailyLow22)) | (np.isnan(dailyLow21) & np.isnan(dailyLow22))) &
        (np.greater(dailyLow11[3:], dailyLow22[3:], where=~np.isnan(dailyLow11[3:]) & ~np.isnan(dailyLow22[3:])) | (np.isnan(dailyLow11[3:]) & np.isnan(dailyLow22[3:]))),
        False
    )

    # Forward fill lastSwingType: 1=dailyHigh, 2=dailyLow
    swing_type_mask = np.zeros(n, dtype=np.int32)
    swing_type_mask[is_swing_high_mask] = 1
    swing_type_mask[is_swing_low_mask] = 2

    # Fill forward: track last non-zero value
    cumsum = np.cumsum(swing_type_mask != 0)
    nonzero_mask = swing_type_mask != 0
    first_nonzero_idx = np.argmax(nonzero_mask) if np.any(nonzero_mask) else n
    if first_nonzero_idx < n:
        swing_type_mask[:first_nonzero_idx + 1] = swing_type_mask[first_nonzero_idx]
    swing_type_mask = np.maximum.accumulate(swing_type_mask)

    # Build boolean Series
    bfvg_condition = pd.Series(np.zeros(n, dtype=bool))
    sfvg_condition = pd.Series(np.zeros(n, dtype=bool))
    isWithinWindow = pd.Series(np.zeros(n, dtype=bool))
    barstate_isconfirmed = pd.Series(np.ones(n, dtype=bool))

    # Map conditions to bars
    if len(df_4h) >= 3:
        for i in range(2, len(df_4h)):
            bar_idx = i
            if bar_idx < n:
                bfvg_condition.iloc[bar_idx] = (low_4h[i] > high_4h_2[i])
                sfvg_condition.iloc[bar_idx] = (high_4h[i] < low_4h_2[i])

    # Trading windows (London time)
    ts_arr = df['time'].values
    for i in range(n):
        if np.isnan(ts_arr[i]):
            continue
        bar_time = datetime.fromtimestamp(ts_arr[i], tz=timezone.utc)
        hour = bar_time.hour
        minute = bar_time.minute
        total_minutes = hour * 60 + minute
        in_w1 = (total_minutes >= 7 * 60 + 45) and (total_minutes < 11 * 60 + 45)
        in_w2 = (total_minutes >= 14 * 60) and (total_minutes < 14 * 60 + 45)
        isWithinWindow.iloc[i] = in_w1 or in_w2

    # Filter: barstate.isfirst or barstate.isconfirmed -> all True (no lookahead)
    isRealTimeOrConfirmed = barstate_isconfirmed

    # Build condition series
    bullish_entry = bfvg_condition & isRealTimeOrConfirmed & (swing_type_mask == 2) & isWithinWindow
    bearish_entry = sfvg_condition & isRealTimeOrConfirmed & (swing_type_mask == 1) & isWithinWindow

    entries = []
    trade_num = 1
    for i in range(n):
        if bullish_entry.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif bearish_entry.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1

    return entries