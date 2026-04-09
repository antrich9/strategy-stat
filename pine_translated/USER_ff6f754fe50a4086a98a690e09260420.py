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
    
    # Time window filtering - London sessions
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    
    isWithinMorningWindow = (df['hour'] == 7) & (df['minute'] >= 45) | (df['hour'] == 8) | (df['hour'] == 9) & (df['minute'] < 45)
    isWithinAfternoonWindow = (df['hour'] == 14) & (df['minute'] >= 45) | (df['hour'] == 15) | (df['hour'] == 16) & (df['minute'] < 45)
    isWithinTimeWindow = isWithinMorningWindow | isWithinAfternoonWindow
    
    # 4H data using 240-minute resampling
    df_4h = df.set_index('dt').resample('240T').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    df_4h = df_4h.reset_index()
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Swing detection on 4H data
    is_swing_high_4h = np.zeros(len(df_4h), dtype=bool)
    is_swing_low_4h = np.zeros(len(df_4h), dtype=bool)
    
    for i in range(5, len(df_4h) - 5):
        if high_4h[i-3] < high_4h[i-2] and high_4h[i-1] <= high_4h[i-2] and high_4h[i-2] >= high_4h[i-4] and high_4h[i-2] >= high_4h[i-5]:
            is_swing_high_4h[i] = True
        if low_4h[i-3] > low_4h[i-2] and low_4h[i-1] >= low_4h[i-2] and low_4h[i-2] <= low_4h[i-4] and low_4h[i-2] <= low_4h[i-5]:
            is_swing_low_4h[i] = True
    
    # FVG detection on 4H data
    bull_fvg_4h = np.zeros(len(df_4h), dtype=bool)
    bear_fvg_4h = np.zeros(len(df_4h), dtype=bool)
    
    for i in range(2, len(df_4h)):
        bull_fvg_4h[i] = close_4h[i-1] < open_4h[i-1] and close_4h[i-2] > open_4h[i-2] and low_4h[i-1] > high_4h[i-2]
        bear_fvg_4h[i] = close_4h[i-1] > open_4h[i-1] and close_4h[i-2] < open_4h[i-2] and high_4h[i-1] < low_4h[i-2]
    
    # Previous day high/low from 4H high/low shifted
    prev_day_high = pd.Series(high_4h).shift(1).fillna(method='bfill').values
    prev_day_low = pd.Series(low_4h).shift(1).fillna(method='bfill').values
    
    # PDH/PDL sweep flags
    pdh_swept = False
    pdl_swept = False
    
    # Map 4H indicators back to main df using merge on time
    df_4h_temp = df_4h.copy()
    df_4h_temp['is_swing_high_4h'] = is_swing_high_4h
    df_4h_temp['is_swing_low_4h'] = is_swing_low_4h
    df_4h_temp['bull_fvg_4h'] = bull_fvg_4h
    df_4h_temp['bear_fvg_4h'] = bear_fvg_4h
    df_4h_temp['prev_day_high'] = prev_day_high
    df_4h_temp['prev_day_low'] = prev_day_low
    df_4h_temp = df_4h_temp.rename(columns={'dt': 'time_dt'})
    
    df_merged = pd.merge_asof(df.sort_values('time'), df_4h_temp.sort_values('time_dt'), left_on='time', right_on=df_4h_temp['time_dt'].astype(np.int64) // 10**9, direction='backward')
    
    # Entry conditions
    long_entry_cond = df_merged['bear_fvg_4h'] & df_merged['is_swing_high_4h'] & isWithinTimeWindow.values[df_merged.index] & (df_merged['close'] > df_merged['prev_day_low']) & (~df_merged.index.isin([df_merged[df_merged['prev_day_low'].notna() & (df_merged['low'] > df_merged['prev_day_low'])].index for _ in range(1)]))
    short_entry_cond = df_merged['bull_fvg_4h'] & df_merged['is_swing_low_4h'] & isWithinTimeWindow.values[df_merged.index] & (df_merged['close'] < df_merged['prev_day_high']) & (~df_merged.index.isin([df_merged[df_merged['prev_day_high'].notna() & (df_merged['high'] < df_merged['prev_day_high'])].index for _ in range(1)]))
    
    for i in range(len(df_merged)):
        if i == 0:
            continue
        if pd.isna(df_merged['bear_fvg_4h'].iloc[i]) or pd.isna(df_merged['is_swing_high_4h'].iloc[i]):
            continue
        
        if i > 0 and not pd.isna(df_merged['high'].iloc[i]) and not pd.isna(df_merged['prev_day_high'].iloc[i]):
            if df_merged['high'].iloc[i] > df_merged['prev_day_high'].iloc[i]:
                pdh_swept = True
                pdl_swept = False
            elif df_merged['low'].iloc[i] < df_merged['prev_day_low'].iloc[i]:
                pdl_swept = True
                pdh_swept = False
        
        entry_price = df_merged['close'].iloc[i]
        ts = int(df_merged['time'].iloc[i])
        
        if df_merged['bear_fvg_4h'].iloc[i] and df_merged['is_swing_high_4h'].iloc[i] and bool(isWithinTimeWindow.iloc[i] if i < len(isWithinTimeWindow) else False) and df_merged['close'].iloc[i] > df_merged['prev_day_low'].iloc[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif df_merged['bull_fvg_4h'].iloc[i] and df_merged['is_swing_low_4h'].iloc[i] and bool(isWithinTimeWindow.iloc[i] if i < len(isWithinTimeWindow) else False) and df_merged['close'].iloc[i] < df_merged['prev_day_high'].iloc[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return results