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
    trade_num = 0
    
    time_col = df['time']
    open_col = df['open']
    high_col = df['high']
    low_col = df['low']
    close_col = df['close']
    volume_col = df['volume']
    
    lookback = 3
    inp11 = False
    inp21 = False
    inp31 = False
    
    london_start_window1 = time_col.apply(lambda x: datetime(x // 1000 if x > 1e12 else x, tz=timezone.utc).replace(hour=7, minute=45, second=0, microsecond=0).timestamp() * 1000 if x > 1e12 else x)
    london_end_window1 = time_col.apply(lambda x: datetime(x // 1000 if x > 1e12 else x, tz=timezone.utc).replace(hour=11, minute=45, second=0, microsecond=0).timestamp() * 1000 if x > 1e12 else x)
    london_start_window2 = time_col.apply(lambda x: datetime(x // 1000 if x > 1e12 else x, tz=timezone.utc).replace(hour=14, minute=0, second=0, microsecond=0).timestamp() * 1000 if x > 1e12 else x)
    london_end_window2 = time_col.apply(lambda x: datetime(x // 1000 if x > 1e12 else x, tz=timezone.utc).replace(hour=14, minute=45, second=0, microsecond=0).timestamp() * 1000 if x > 1e12 else x)
    
    isWithinWindow1 = (time_col >= london_start_window1) & (time_col < london_end_window1)
    isWithinWindow2 = (time_col >= london_start_window2) & (time_col < london_end_window2)
    in_trading_window = isWithinWindow1 | isWithinWindow2
    
    def is_new_4h_bar(idx):
        if idx < 1:
            return False
        curr_time = time_col.iloc[idx]
        prev_time = time_col.iloc[idx - 1]
        curr_dt = datetime.fromtimestamp(curr_time / 1000 if curr_time > 1e12 else curr_time, tz=timezone.utc)
        prev_dt = datetime.fromtimestamp(prev_time / 1000 if prev_time > 1e12 else prev_time, tz=timezone.utc)
        curr_4h = (curr_dt.hour // 4)
        prev_4h = (prev_dt.hour // 4)
        curr_day = curr_dt.day
        prev_day = prev_dt.day
        return curr_4h != prev_4h or curr_day != prev_day
    
    volume_4h = volume_col.rolling(4, min_periods=4).sum()
    volfilt1 = inp11 & (volume_4h.shift(1) > volume_4h.rolling(9).mean().shift(1) * 1.5)
    
    atr_len = 20
    high_low = high_col - low_col
    high_close = (high_col - close_col.shift(1)).abs()
    low_close = (low_col - close_col.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/atr_len, adjust=False).mean()
    vol_atr_filt = pd.Series(True, index=df.index)
    if inp21:
        vol_atr_filt = atr.shift(1) > 0
    
    close_arr = close_col.values
    high_arr = high_col.values
    low_arr = low_col.values
    close_shifted = np.roll(close_arr, 1)
    high_shifted = np.roll(high_arr, 1)
    low_shifted = np.roll(low_arr, 1)
    close_shifted[0] = np.nan
    high_shifted[0] = np.nan
    low_shifted[0] = np.nan
    
    phC = np.zeros(len(df)) * np.nan
    plC = np.zeros(len(df)) * np.nan
    for i in range(lookback, len(df)):
        left_start = i - lookback - 1
        right_end = i - lookback
        if left_start >= 0:
            highs_in_window = high_arr[left_start:right_end]
            lows_in_window = low_arr[left_start:right_end]
            if len(highs_in_window) > 0:
                max_high_idx = np.argmax(highs_in_window)
                min_low_idx = np.argmin(lows_in_window)
                if max_high_idx == len(highs_in_window) - 1:
                    phC[i] = high_arr[right_end]
                if min_low_idx == len(lows_in_window) - 1:
                    plC[i] = low_arr[right_end]
    
    phC = pd.Series(phC, index=df.index)
    plC = pd.Series(plC, index=df.index)
    
    fvg_bullish = np.zeros(len(df), dtype=bool)
    fvg_bearish = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)):
        if i >= 2:
            bull_cond = low_arr[i] > high_arr[i-2] and close_arr[i-1] > close_arr[i-2] and close_arr[i] > close_arr[i-1]
            bear_cond = high_arr[i] < low_arr[i-2] and close_arr[i-1] < close_arr[i-2] and close_arr[i] < close_arr[i-1]
            if bull_cond:
                fvg_bullish[i] = True
            if bear_cond:
                fvg_bearish[i] = True
    
    fvg_bullish = pd.Series(fvg_bullish, index=df.index)
    fvg_bearish = pd.Series(fvg_bearish, index=df.index)
    
    in_trend_up_arr = np.zeros(len(df), dtype=bool)
    in_trend_down_arr = np.zeros(len(df), dtype=bool)
    in_trend_up_arr[0] = False
    in_trend_down_arr[0] = True
    for i in range(1, len(df)):
        bull_broken = high_arr[i] > close_shifted[i]
        bear_broken = low_arr[i] < close_shifted[i]
        if bull_broken:
            in_trend_up_arr[i] = True
            in_trend_down_arr[i] = False
        elif bear_broken:
            in_trend_up_arr[i] = False
            in_trend_down_arr[i] = True
        else:
            in_trend_up_arr[i] = in_trend_up_arr[i-1]
            in_trend_down_arr[i] = in_trend_down_arr[i-1]
    
    in_trend_up = pd.Series(in_trend_up_arr, index=df.index)
    in_trend_down = pd.Series(in_trend_down_arr, index=df.index)
    
    in_trend_filt = pd.Series(True, index=df.index)
    if inp31:
        in_trend_filt = in_trend_up
    
    ext1 = 10
    
    pivot_high_broken = np.zeros(len(df), dtype=bool)
    pivot_low_broken = np.zeros(len(df), dtype=bool)
    for i in range(lookback + 1, len(df)):
        if not np.isnan(phC.iloc[i]) and phC.iloc[i] > 0:
            if high_arr[i] > phC.iloc[i]:
                pivot_high_broken[i] = True
        if not np.isnan(plC.iloc[i]) and plC.iloc[i] > 0:
            if low_arr[i] < plC.iloc[i]:
                pivot_low_broken[i] = True
    
    pivot_high_broken = pd.Series(pivot_high_broken, index=df.index)
    pivot_low_broken = pd.Series(pivot_low_broken, index=df.index)
    
    for i in range(ext1 + 2, len(df)):
        if pd.isna(close_col.iloc[i]) or pd.isna(high_col.iloc[i]) or pd.isna(low_col.iloc[i]):
            continue
        
        if not in_trading_window.iloc[i]:
            continue
        
        bull_fvg_near = fvg_bullish.iloc[max(0, i-ext1):i+1].any()
        bear_fvg_near = fvg_bearish.iloc[max(0, i-ext1):i+1].any()
        
        bull_conditions = (
            pivot_high_broken.iloc[i] and
            bull_fvg_near and
            volfilt1.iloc[i] if inp11 else True and
            vol_atr_filt.iloc[i] and
            in_trend_filt.iloc[i]
        )
        
        bear_conditions = (
            pivot_low_broken.iloc[i] and
            bear_fvg_near and
            volfilt1.iloc[i] if inp11 else True and
            vol_atr_filt.iloc[i] and
            in_trend_filt.iloc[i]
        )
        
        if bull_conditions:
            trade_num += 1
            ts = int(time_col.iloc[i])
            entry_price = float(close_col.iloc[i])
            entry_time_str = datetime.fromtimestamp(ts / 1000 if ts > 1e12 else ts, tz=timezone.utc).isoformat()
            results.append({
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
        
        if bear_conditions:
            trade_num += 1
            ts = int(time_col.iloc[i])
            entry_price = float(close_col.iloc[i])
            entry_time_str = datetime.fromtimestamp(ts / 1000 if ts > 1e12 else ts, tz=timezone.utc).isoformat()
            results.append({
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
    
    return results