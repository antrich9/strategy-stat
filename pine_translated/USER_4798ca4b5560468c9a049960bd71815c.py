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
    
    # Calculate current timeframe EMAs (9 and 18)
    cema9 = df['close'].ewm(span=9, adjust=False).mean()
    cema18 = df['close'].ewm(span=18, adjust=False).mean()
    
    # Calculate daily EMAs using request.security equivalent
    # Convert to datetime for resampling
    df_copy = df.copy()
    df_copy['datetime'] = pd.to_datetime(df_copy['time'], unit='s', utc=True)
    df_copy.set_index('datetime', inplace=True)
    
    # Resample to daily and calculate EMAs
    daily_df = df_copy.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    daily_ema9 = daily_df['close'].ewm(span=9, adjust=False).mean()
    daily_ema18 = daily_df['close'].ewm(span=18, adjust=False).mean()
    
    # Map daily EMAs back to original timeframe using forward fill
    df_copy['ema9'] = daily_ema9.reindex(df_copy.index, method='ffill')
    df_copy['ema18'] = daily_ema18.reindex(df_copy.index, method='ffill')
    
    ema9 = df_copy['ema9'].values
    ema18 = df_copy['ema18'].values
    
    # Get previous day's high and low
    daily_high = daily_df['high'].shift(1)
    daily_low = daily_df['low'].shift(1)
    
    df_copy['prevDayHigh'] = daily_high.reindex(df_copy.index, method='ffill').values
    df_copy['prevDayLow'] = daily_low.reindex(df_copy.index, method='ffill').values
    
    # Detect new day
    df_copy['date'] = df_copy.index.date
    df_copy['isNewDay'] = df_copy['date'] != df_copy['date'].shift(1)
    
    # Flags for previous day high/low sweep (reset on new day)
    flagpdh = False
    flagpdl = False
    
    # Previous day high and low values (updated on new day)
    prevDayHigh = np.nan
    prevDayLow = np.nan
    
    # Time filter logic
    df_copy['hour'] = df_copy.index.hour
    df_copy['minute'] = df_copy.index.minute
    df_copy['time_minutes'] = df_copy['hour'] * 60 + df_copy['minute']
    
    # Session 1: 07:00-09:59 (420-599 minutes)
    # Session 2: 12:00-14:59 (720-899 minutes)
    is_in_session1 = (df_copy['time_minutes'] >= 420) & (df_copy['time_minutes'] <= 599)
    is_in_session2 = (df_copy['time_minutes'] >= 720) & (df_copy['time_minutes'] <= 899)
    is_in_session = is_in_session1 | is_in_session2
    
    # OB and FVG conditions
    close = df['close'].values
    open_prices = df['open'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    
    # Helper functions for OB/FVG
    def is_up(idx):
        if idx < 0 or idx >= len(close):
            return False
        return close[idx] > open_prices[idx]
    
    def is_down(idx):
        if idx < 0 or idx >= len(close):
            return False
        return close[idx] < open_prices[idx]
    
    def is_ob_up(idx):
        if idx < 0 or idx + 1 >= len(close):
            return False
        return is_down(idx + 1) and is_up(idx) and close[idx] > high_prices[idx + 1]
    
    def is_ob_down(idx):
        if idx < 0 or idx + 1 >= len(close):
            return False
        return is_up(idx + 1) and is_down(idx) and close[idx] < low_prices[idx + 1]
    
    def is_fvg_up(idx):
        if idx < 0 or idx + 2 >= len(close):
            return False
        return low_prices[idx] > high_prices[idx + 2]
    
    def is_fvg_down(idx):
        if idx < 0 or idx + 2 >= len(close):
            return False
        return high_prices[idx] < low_prices[idx + 2]
    
    # Main conditions
    condition_long = (close > ema9) & (close > ema18) & (close > cema9.values) & (close > cema18.values)
    condition_short = (close < ema9) & (close < ema18) & (close < cema9.values) & (close < cema18.values)
    
    # Iterate through bars
    for i in range(1, len(df)):
        ts = int(df['time'].iloc[i])
        
        # Update previous day high/low on new day
        if df_copy['isNewDay'].iloc[i]:
            flagpdh = False
            flagpdl = False
            prevDayHigh = df_copy['prevDayHigh'].iloc[i]
            prevDayLow = df_copy['prevDayLow'].iloc[i]
        
        # Check for sweep of previous day high
        if close[i] > prevDayHigh and not np.isnan(prevDayHigh):
            flagpdh = True
        
        # Check for sweep of previous day low
        if close[i] < prevDayLow and not np.isnan(prevDayLow):
            flagpdl = True
        
        # Check time filter
        in_session = is_in_session.iloc[i]
        
        # Calculate OB/FVG for current bar (index i, reference bar 1 and 0)
        ob_up_current = is_ob_up(1)  # relative to bar 1
        fvg_up_current = is_fvg_up(0)
        ob_down_current = is_ob_down(1)
        fvg_down_current = is_fvg_down(0)
        
        # Long entry: condition_long AND flagpdh AND in_session AND (ob_up AND fvg_up)
        if condition_long[i] and flagpdh and in_session and ob_up_current and fvg_up_current:
            trade_num += 1
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': close[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close[i],
                'raw_price_b': close[i]
            })
        
        # Short entry: condition_short AND flagpdl AND in_session AND (ob_down AND fvg_down)
        if condition_short[i] and flagpdl and in_session and ob_down_current and fvg_down_current:
            trade_num += 1
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': close[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close[i],
                'raw_price_b': close[i]
            })
    
    return results