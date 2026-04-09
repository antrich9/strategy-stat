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
    
    open_series = df['open']
    high_series = df['high']
    low_series = df['low']
    close_series = df['close']
    time_series = df['time']
    
    # London trading windows (7:45-9:45 and 15:45-16:45)
    timestamps = pd.to_datetime(time_series, unit='s', utc=True)
    hours = timestamps.hour + timestamps.minute / 60.0
    
    morning_window = (hours >= 7.75) & (hours < 9.75)
    afternoon_window = (hours >= 15.75) & (hours < 16.75)
    in_trading_window = morning_window | afternoon_window
    
    # ATR(200)
    tr1 = high_series - low_series
    tr2 = np.abs(high_series - close_series.shift(1))
    tr3 = np.abs(low_series - close_series.shift(1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = true_range.ewm(span=200, adjust=False).mean()
    
    # Previous day high/low detection
    day_change = (timestamps.diff().dt.days < 0) | timestamps.diff().dt.days.isna()
    is_new_day = day_change.fillna(True)
    
    prev_day_high = high_series.rolling(window=2, min_periods=2).apply(lambda x: x.iloc[0] if len(x) >= 2 else np.nan, raw=True).shift(1)
    prev_day_low = low_series.rolling(window=2, min_periods=2).apply(lambda x: x.iloc[0] if len(x) >= 2 else np.nan, raw=True).shift(1)
    
    # Current 4H high/low
    current_4h_high = high_series.rolling(window=4, min_periods=1).max()
    current_4h_low = low_series.rolling(window=4, min_periods=1).min()
    
    # Previous day high/low taken
    prev_high_taken = high_series > prev_day_high
    prev_low_taken = low_series < prev_day_low
    
    # Flags
    flagpdh = np.where(
        (prev_high_taken) & (current_4h_low > prev_day_low),
        True,
        np.where(
            (prev_low_taken) & (current_4h_high < prev_day_high),
            False,
            np.nan
        )
    )
    flagpdl = np.where(
        (prev_low_taken) & (current_4h_high < prev_day_high),
        True,
        np.where(
            (prev_high_taken) & (current_4h_low > prev_day_low),
            False,
            np.nan
        )
    )
    
    # Forward-fill flags
    flagpdh_series = pd.Series(flagpdh).ffill()
    flagpdl_series = pd.Series(flagpdl).ffill()
    
    # FVG Detection
    bull_gap_top = np.minimum(close_series, open_series)
    bull_gap_btm = np.maximum(close_series.shift(1), open_series.shift(1))
    
    bull_fvg_cond = (open_series > close_series.shift(1)) & (close_series > close_series.shift(1)) & (close_series.shift(1) > open_series.shift(1))
    
    bull_fvg = bull_fvg_cond & (close_series.shift(1) < bull_gap_top) & (bull_gap_btm < bull_gap_top)
    
    bear_gap_top = np.maximum(close_series.shift(1), open_series.shift(1))
    bear_gap_btm = np.minimum(close_series, open_series)
    
    bear_fvg_cond = (open_series < close_series.shift(1)) & (close_series < close_series.shift(1)) & (close_series.shift(1) < open_series.shift(1))
    
    bear_fvg = bear_fvg_cond & (close_series.shift(1) > bear_gap_btm) & (bear_gap_top > bear_gap_btm)
    
    # Volume Imbalances
    bull_vi_top = np.minimum(close_series, open_series)
    bull_vi_btm = np.maximum(close_series.shift(1), open_series.shift(1))
    
    bull_vi_cond = (open_series > close_series.shift(1)) & (high_series.shift(1) > low_series) & (close_series > close_series.shift(1)) & (open_series > open_series.shift(1)) & (high_series.shift(1) < bull_vi_top)
    
    bull_vi = bull_vi_cond
    
    bear_vi_top = np.minimum(close_series.shift(1), open_series.shift(1))
    bear_vi_btm = np.maximum(close_series, open_series)
    
    bear_vi_cond = (open_series < close_series.shift(1)) & (high_series.shift(1) > low_series) & (close_series < close_series.shift(1)) & (open_series < open_series.shift(1)) & (low_series.shift(1) < bear_vi_btm)
    
    bear_vi = bear_vi_cond
    
    # Long Entry: Trading window AND (flagpdh OR bull_fvg) AND NOT flagpdl
    # Short Entry: Trading window AND (flagpdl OR bear_fvg) AND NOT flagpdh
    for i in range(len(df)):
        if i < 2:
            continue
        if in_trading_window.iloc[i] and not pd.isna(atr.iloc[i]) and not pd.isna(prev_day_high.iloc[i]) and not pd.isna(prev_day_low.iloc[i]):
            # Long conditions
            long_cond = (in_trading_window.iloc[i] and 
                        ((flagpdh_series.iloc[i] == True) or (bull_fvg.iloc[i] == True)) and
                        (flagpdl_series.iloc[i] != True))
            
            # Short conditions
            short_cond = (in_trading_window.iloc[i] and 
                         ((flagpdl_series.iloc[i] == True) or (bear_fvg.iloc[i] == True)) and
                         (flagpdh_series.iloc[i] != True))
            
            if long_cond:
                ts = int(time_series.iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': close_series.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close_series.iloc[i],
                    'raw_price_b': close_series.iloc[i]
                })
                trade_num += 1
            elif short_cond:
                ts = int(time_series.iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': close_series.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close_series.iloc[i],
                    'raw_price_b': close_series.iloc[i]
                })
                trade_num += 1
    
    return entries