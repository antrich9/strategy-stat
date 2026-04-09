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
    trades = []
    trade_num = 1

    if len(df) < 10:
        return trades

    time_col = df['time']
    high_col = df['high']
    low_col = df['low']
    close_col = df['close']
    open_col = df['open']

    london_tz = timezone.utc

    def is_in_trading_window(ts):
        dt = datetime.fromtimestamp(ts, tz=london_tz)
        hour = dt.hour
        minute = dt.minute
        total_mins = hour * 60 + minute
        morning_start = 7 * 60 + 45
        morning_end = 9 * 60 + 45
        afternoon_start = 14 * 60 + 45
        afternoon_end = 16 * 60 + 45
        return (morning_start <= total_mins < morning_end) or (afternoon_start <= total_mins < afternoon_end)

    is_within_time_window = df['time'].apply(is_in_trading_window)

    timeframe_4h_secs = 240 * 60
    df_4h = df.copy()
    df_4h['time_4h'] = (df_4h['time'] // timeframe_4h_secs) * timeframe_4h_secs
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'time': 'first'}
    df_4h_agg = df_4h.groupby('time_4h').agg(agg_dict).reset_index()

    high_4h = df_4h_agg['high'].values
    low_4h = df_4h_agg['low'].values

    is_swing_high_4h = np.zeros(len(high_4h), dtype=bool)
    is_swing_low_4h = np.zeros(len(low_4h), dtype=bool)

    for i in range(5, len(high_4h)):
        if (high_4h[i-3] < high_4h[i-2] and
            high_4h[i-1] <= high_4h[i-2] and
            high_4h[i-2] >= high_4h[i-4] and
            high_4h[i-2] >= high_4h[i-5]):
            is_swing_high_4h[i] = True
        if (low_4h[i-3] > low_4h[i-2] and
            low_4h[i-1] >= low_4h[i-2] and
            low_4h[i-2] <= low_4h[i-4] and
            low_4h[i-2] <= low_4h[i-5]):
            is_swing_low_4h[i] = True

    last_swing_high_4h = np.full(len(df), np.nan)
    last_swing_low_4h = np.full(len(df), np.nan)
    trend_direction = np.full(len(df), 'Neutral', dtype=object)
    bullish_count = np.zeros(len(df), dtype=int)
    bearish_count = np.zeros(len(df), dtype=int)

    last_sh = np.nan
    last_sl = np.nan
    bull_cnt = 0
    bear_cnt = 0
    last_4h_idx = -1

    for i in range(len(df)):
        current_time = df['time'].iloc[i]
        current_4h_ts = (current_time // timeframe_4h_secs) * timeframe_4h_secs
        idx_4h = df_4h_agg[df_4h_agg['time_4h'] == current_4h_ts].index
        if len(idx_4h) > 0:
            idx_4h = idx_4h[0]
            if idx_4h != last_4h_idx and idx_4h >= 5:
                if is_swing_high_4h[idx_4h]:
                    last_sh = high_4h[idx_4h]
                    bull_cnt += 1
                    bear_cnt = 0
                if is_swing_low_4h[idx_4h]:
                    last_sl = low_4h[idx_4h]
                    bear_cnt += 1
                    bull_cnt = 0
                last_4h_idx = idx_4h
        last_swing_high_4h[i] = last_sh
        last_swing_low_4h[i] = last_sl
        if bull_cnt > 1:
            trend_direction[i] = 'Bullish'
        elif bear_cnt > 1:
            trend_direction[i] = 'Bearish'

    df['prev_day_high'] = np.nan
    df['prev_day_low'] = np.nan
    df['pdh_swept'] = False
    df['pdl_swept'] = False

    for i in range(len(df)):
        current_dt = datetime.fromtimestamp(df['time'].iloc[i], tz=london_tz)
        prev_day_start_ts = int(datetime(current_dt.year, current_dt.month, current_dt.day, 0, 0, tzinfo=london_tz).timestamp()) - 86400
        prev_day_end_ts = prev_day_start_ts + 86400
        mask_prev_day = (df['time'] >= prev_day_start_ts) & (df['time'] < prev_day_end_ts)
        if mask_prev_day.any():
            df.loc[mask_prev_day, 'prev_day_high'] = df.loc[mask_prev_day, 'high'].max()
            df.loc[mask_prev_day, 'prev_day_low'] = df.loc[mask_prev_day, 'low'].min()

    for i in range(1, len(df)):
        if df['high'].iloc[i] > df['prev_day_high'].iloc[i]:
            df.at[df.index[i], 'pdh_swept'] = True
        if df['low'].iloc[i] < df['prev_day_low'].iloc[i]:
            df.at[df.index[i], 'pdl_swept'] = True

    df['is_bullish_fvg'] = False
    df['is_bearish_fvg'] = False

    for i in range(2, len(df)):
        if i < 3:
            continue
        low1 = low_col.iloc[i-2]
        high0 = high_col.iloc[i]
        high1 = high_col.iloc[i-2]
        low0 = low_col.iloc[i]
        if high0 < low1 and low0 > high1:
            fvg_mid = (low1 + high1) / 2.0
            if close_col.iloc[i] > fvg_mid:
                df.at[df.index[i], 'is_bullish_fvg'] = True
            if close_col.iloc[i] < fvg_mid:
                df.at[df.index[i], 'is_bearish_fvg'] = True

    bull_fvg_upper = np.full(len(df), np.nan)
    bull_fvg_lower = np.full(len(df), np.nan)
    bear_fvg_upper = np.full(len(df), np.nan)
    bear_fvg_lower = np.full(len(df), np.nan)
    bull_fvg_active = False
    bear_fvg_active = False
    bull_fvg_up_val = np.nan
    bull_fvg_lo_val = np.nan
    bear_fvg_up_val = np.nan
    bear_fvg_lo_val = np.nan

    for i in range(2, len(df)):
        low1 = low_col.iloc[i-2]
        high1 = high_col.iloc[i-2]
        high0 = high_col.iloc[i]
        low0 = low_col.iloc[i]
        close_i = close_col.iloc[i]

        if high0 < low1 and low0 > high1:
            if close_i > (low1 + high1) / 2.0:
                bull_fvg_active = True
                bear_fvg_active = False
                bull_fvg_up_val = low1
                bull_fvg_lo_val = high1
            elif close_i < (low1 + high1) / 2.0:
                bear_fvg_active = True
                bull_fvg_active = False
                bear_fvg_up_val = low1
                bear_fvg_lo_val = high1

        if bull_fvg_active:
            if close_i > bull_fvg_lo_val and close_i < bull_fvg_up_val:
                if close_i < bull_fvg_lo_val:
                    pass
            bull_fvg_upper[i] = bull_fvg_up_val
            bull_fvg_lower[i] = bull_fvg_lo_val
        elif bear_fvg_active:
            if close_i < bear_fvg_up_val and close_i > bear_fvg_lo_val:
                if close_i > bear_fvg_up_val:
                    pass
            bear_fvg_upper[i] = bear_fvg_up_val
            bear_fvg_lower[i] = bear_fvg_lo_val

    entry_long_condition = (
        is_within_time_window &
        (trend_direction == 'Bullish') &
        df['is_bullish_fvg']
    )

    entry_short_condition = (
        is_within_time_window &
        (trend_direction == 'Bearish') &
        df['is_bearish_fvg']
    )

    has_long = entry_long_condition.any()
    has_short = entry_short_condition.any()

    for i in range(len(df)):
        if has_long and entry_long_condition.iloc[i]:
            entry_price = close_col.iloc[i]
            ts = int(time_col.iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=london_tz).isoformat()
            trades.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    for i in range(len(df)):
        if has_short and entry_short_condition.iloc[i]:
            entry_price = close_col.iloc[i]
            ts = int(time_col.iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=london_tz).isoformat()
            trades.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return trades