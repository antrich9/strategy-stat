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
    high = df['high']
    low = df['low']
    close = df['close']
    open_ = df['open']
    volume = df['volume']
    time = df['time']

    entries = []
    trade_num = 1

    # Swing detection
    is_swing_high = pd.Series(False, index=df.index)
    is_swing_low = pd.Series(False, index=df.index)

    main_bar_high = high.shift(2)
    main_bar_low = low.shift(2)

    condition_high = (high.shift(1) < main_bar_high) & (high.shift(3) < main_bar_high) & (high.shift(4) < main_bar_high)
    condition_low = (low.shift(1) > main_bar_low) & (low.shift(3) > main_bar_low) & (low.shift(4) > main_bar_low)

    for i in range(5, len(df)):
        if condition_high.iloc[i]:
            is_swing_high.iloc[i] = True
        if condition_low.iloc[i]:
            is_swing_low.iloc[i] = True

    # Store last swing high and low
    last_swing_high = pd.Series(np.nan, index=df.index)
    last_swing_low = pd.Series(np.nan, index=df.index)

    for i in range(len(df)):
        if is_swing_high.iloc[i]:
            last_swing_high.iloc[i] = main_bar_high.iloc[i]
        elif i > 0 and not pd.isna(last_swing_high.iloc[i-1]):
            last_swing_high.iloc[i] = last_swing_high.iloc[i-1]

        if is_swing_low.iloc[i]:
            last_swing_low.iloc[i] = main_bar_low.iloc[i]
        elif i > 0 and not pd.isna(last_swing_low.iloc[i-1]):
            last_swing_low.iloc[i] = last_swing_low.iloc[i-1]

    # Time window (London time 07:45-09:45 and 14:45-16:45)
    times = pd.to_datetime(df['time'], unit='s', utc=True)

    def in_london_window(hour, minute):
        return (hour == 7 and minute >= 45) or (hour == 8) or (hour == 9 and minute <= 45) or \
               (hour == 14 and minute >= 45) or (hour == 15) or (hour == 16 and minute <= 45)

    is_within_time_window = pd.Series(False, index=df.index)
    for i in range(len(df)):
        t = times.iloc[i].to_pydatetime()
        if in_london_window(t.hour, t.minute):
            is_within_time_window.iloc[i] = True

    # Volume filter
    vol_sma = volume.rolling(9).mean()
    vol_filt = volume.shift(1) > vol_sma * 1.5

    # ATR filter
    atr = (high - low).rolling(20).max() / 1.5

    # Trend filter
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    loc_filt_b = loc2
    loc_filt_s = ~loc2

    # FVG conditions
    bfvg = (low > high.shift(2)) & vol_filt & (atr > 0) & loc_filt_b
    sfvg = (high < low.shift(2)) & vol_filt & (atr > 0) & loc_filt_s

    # Prev day high/low (use 1 day lookback)
    prev_day_high = high.shift(1)
    prev_day_low = low.shift(1)

    # Sweep conditions
    sweep_bull_high = close > prev_day_high
    sweep_bear_low = close < prev_day_low

    # Long entry conditions
    long_cond = bfvg & is_within_time_window

    # Short entry conditions
    short_cond = sfvg & is_within_time_window

    # Generate entries
    for i in range(5, len(df)):
        if long_cond.iloc[i] and not pd.isna(close.iloc[i]):
            entry_ts = int(time.iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

        if short_cond.iloc[i] and not pd.isna(close.iloc[i]):
            entry_ts = int(time.iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

    return entries