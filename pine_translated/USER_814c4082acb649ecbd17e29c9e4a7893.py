import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

def generate_entries(df: pd.DataFrame) -> list:
    if len(df) < 50:
        return []
    
    high = df['high']
    low = df['low']
    close = df['close']
    open_price = df['open']
    volume = df['volume']
    time = df['time']
    
    # ATR calculation (Wilder's method, 14 period)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # EMA 50
    ema50 = close.ewm(span=50, adjust=False).mean()
    
    # Previous day high and low from 4H data
    df_ts = pd.to_datetime(df['time'], unit='s', utc=True)
    df_indexed = pd.DataFrame({'high': high, 'low': low, 'close': close, 'open': open_price}, index=df_ts)
    resampled_4h = df_indexed.resample('4H').agg({'high': 'max', 'low': 'min'})
    
    prev_day_high = resampled_4h['high'].shift(1)
    prev_day_low = resampled_4h['low'].shift(1)
    
    # Reindex 4H data back to original timeframe using forward fill
    prev_day_high_main = prev_day_high.reindex(df_ts, method='ffill')
    prev_day_low_main = prev_day_low.reindex(df_ts, method='ffill')
    
    # Detect if a bar's high > prev_day_high or low < prev_day_low
    pdh_sweep = high > prev_day_high_main
    pdl_sweep = low < prev_day_low_main
    
    # Swing detection (for reference, not directly used in entries)
    is_swing_high = (high.shift(2) > high.shift(3)) & (high.shift(2) > high.shift(1))
    is_swing_low = (low.shift(2) < low.shift(3)) & (low.shift(2) < low.shift(1))
    
    # FVG detection
    fvg_up = low > high.shift(2)
    fvg_down = high < low.shift(2)
    
    # OB detection
    ob_up = (close.shift(1) < open_price.shift(1)) & (close > open_price) & (close > high.shift(1))
    ob_down = (close.shift(1) > open_price.shift(1)) & (close < open_price) & (close < low.shift(1))
    
    # Time window logic
    ts_dt = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Determine BST (UTC+1) or GMT (UTC) based on date
    def get_utc_offset(dt):
        year = dt.year
        # Last Sunday of March
        march_last = datetime(year, 3, 31, tzinfo=timezone.utc)
        while march_last.weekday() != 6:
            march_last -= timedelta(days=1)
        # Last Sunday of October
        oct_last = datetime(year, 10, 31, tzinfo=timezone.utc)
        while oct_last.weekday() != 6:
            oct_last -= timedelta(days=1)
        if dt >= march_last and dt < oct_last:
            return 1
        return 0
    
    utc_offsets = ts_dt.map(get_utc_offset)
    london_hour = ts_dt.hour + utc_offsets
    
    # Morning window: 10:55 to 14:45
    morning_start = (london_hour > 10) | ((london_hour == 10) & (ts_dt.dt.minute >= 55))
    morning_end = (london_hour < 14) | ((london_hour == 14) & (ts_dt.dt.minute < 45))
    in_morning_window = morning_start & morning_end
    
    # Afternoon window: 14:45 to 18:45
    afternoon_start = (london_hour > 14) | ((london_hour == 14) & (ts_dt.dt.minute >= 45))
    afternoon_end = (london_hour < 18) | ((london_hour == 18) & (ts_dt.dt.minute < 45))
    in_afternoon_window = afternoon_start & afternoon_end
    
    in_trading_window = in_morning_window | in_afternoon_window
    
    # Friday morning restriction
    dayofweek = ts_dt.dt.dayofweek
    is_friday = dayofweek == 4
    is_friday_morning_window = is_friday & in_morning_window
    
    # Filter width check (for FVG)
    filter_width = 0.0
    
    # Build condition series
    bull_fvg_filter = (low.shift(3) - high.shift(1)) > atr * filter_width
    bull_condition = showBull = True
    bull_condition = bull_condition & (low.shift(3) > high.shift(1))
    bull_condition = bull_condition & (close.shift(2) < low.shift(3))
    bull_condition = bull_condition & (close > low.shift(3))
    bull_condition = bull_condition & bull_fvg_filter
    
    bear_fvg_filter = (low.shift(1) - high.shift(3)) > atr * filter_width
    bear_condition = showBear = True
    bear_condition = bear_condition & (low.shift(1) > high.shift(3))
    bear_condition = bear_condition & (close.shift(2) > high.shift(3))
    bear_condition = bear_condition & (close < high.shift(3))
    bear_condition = bear_condition & bear_fvg_filter
    
    # Track trend (bull/bear)
    trend = pd.Series(0, index=df.index)
    bull_mask = bull_condition.fillna(False)
    bear_mask = bear_condition.fillna(False)
    
    for i in range(len(df)):
        if bull_mask.iloc[i]:
            trend.iloc[i] = 1
        elif bear_mask.iloc[i]:
            trend.iloc[i] = 0
        elif i > 0:
            trend.iloc[i] = trend.iloc[i-1]
    
    # Track PDH/PDL flags
    flagpdh = pd.Series(False, index=df.index)
    flagpdl = pd.Series(False, index=df.index)
    
    for i in range(len(df)):
        if i < 4:
            continue
        pdh_val = pdh_sweep.iloc[i] if not pd.isna(pdh_sweep.iloc[i]) else False
        pdl_val = pdl_sweep.iloc[i] if not pd.isna(pdl_sweep.iloc[i]) else False
        
        if pdh_val:
            flagpdh.iloc[i] = True
            flagpdl.iloc[i] = False
        elif pdl_val:
            flagpdl.iloc[i] = True
            flagpdh.iloc[i] = False
        elif i > 0:
            flagpdh.iloc[i] = flagpdh.iloc[i-1]
            flagpdl.iloc[i] = flagpdl.iloc[i-1]
    
    # Entry conditions
    # Long: flagpdl AND fvg_up AND ob_up AND close > ema50
    # (strategy.position_size == 0 AND in_trading_window AND not isFridayMorningWindow)
    
    long_entry = (~flagpdh.shift(1).fillna(False))  # Not currently in short
    long_entry = long_entry & flagpdl
    long_entry = long_entry & fvg_up.fillna(False)
    long_entry = long_entry & ob_up.fillna(False)
    long_entry = long_entry & (close > ema50)
    long_entry = long_entry & in_trading_window
    long_entry = long_entry & (~is_friday_morning_window)
    
    # Generate entries
    entries = []
    trade_num = 0
    
    for i in range(len(df)):
        if i < 5:
            continue
        
        if long_entry.iloc[i]:
            trade_num += 1
            ts = int(time.iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
    
    return entries