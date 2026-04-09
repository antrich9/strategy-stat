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
    
    # Ensure required columns exist
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        return []
    
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # Convert time to datetime for time window filtering
    data['datetime'] = pd.to_datetime(data['time'], unit='s', utc=True)
    data['hour'] = data['datetime'].dt.hour
    data['minute'] = data['datetime'].dt.minute
    data['dayofweek'] = data['datetime'].dt.dayofweek
    
    # Define London time windows (7:45-9:45 and 14:45-16:45)
    morning_window = ((data['hour'] == 7) & (data['minute'] >= 45)) | \
                     ((data['hour'] == 8)) | \
                     ((data['hour'] == 9) & (data['minute'] < 45))
    
    afternoon_window = ((data['hour'] == 14) & (data['minute'] >= 45)) | \
                       ((data['hour'] == 15)) | \
                       ((data['hour'] == 16) & (data['minute'] < 45))
    
    in_trading_window = morning_window | afternoon_window
    
    # Skip weekends (optional, but common in forex)
    weekday_filter = data['dayofweek'] < 5
    in_trading_window = in_trading_window & weekday_filter
    
    # Get 4H (240 min) data for previous day high/low and swing detection
    # In Pine Script, request.security is used to get higher timeframe data
    # We'll simulate this by resampling to 4H and forward-filling
    
    # Create 4H timeframe data
    data_4h = data.set_index('datetime').resample('240T').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'open': 'first',
        'time': 'last'
    }).dropna()
    
    # Forward fill to align 4H data with 15m bars
    data['prev_day_high_4h'] = data['high'].reindex(data_4h.index, method='ffill').reindex(data.index)
    data['prev_day_low_4h'] = data['low'].reindex(data_4h.index, method='ffill').reindex(data.index)
    
    # Get previous day high/low (using 4H high/low as proxy for daily)
    # For proper implementation, we'd need actual daily data
    data['prevDayHigh'] = data['high'].rolling(window=16, min_periods=16).max().shift(1)
    data['prevDayLow'] = data['low'].rolling(window=16, min_periods=16).min().shift(1)
    
    # Detect new day (change in 4H period)
    data['isNewDay'] = (data['datetime'].dt.hour == 0) & (data['datetime'].dt.minute == 0)
    data['isNewDay'] = data['isNewDay'] | (data['datetime'].diff().dt.total_seconds() > 14400)
    
    # Track PDH and PDL sweeps
    data['previousDayHighTaken'] = data['high'] > data['prevDayHigh']
    data['previousDayLowTaken'] = data['low'] < data['prevDayLow']
    
    # Initialize sweep flags (var in Pine = persistent across bars)
    flagpdh = False
    flagpdl = False
    flagpdh_series = pd.Series(False, index=data.index)
    flagpdl_series = pd.Series(False, index=data.index)
    
    for i in range(len(data)):
        if data['high'].iloc[i] > data['prevDayHigh'].iloc[i]:
            flagpdh = True
            flagpdl = False
        elif data['low'].iloc[i] < data['prevDayLow'].iloc[i]:
            flagpdl = True
            flagpdh = False
        flagpdh_series.iloc[i] = flagpdh
        flagpdl_series.iloc[i] = flagpdl
    
    # Swing detection for 4H timeframe
    # In Pine: high_4h[3] < high_4h[2] and high_4h[1] <= high_4h[2] and high_4h[2] >= high_4h[4] and high_4h[2] >= high_4h[5]
    # This means: current high is highest among 5 bars centered at it
    
    high_4h = data_4h['high'].values
    low_4h = data_4h['low'].values
    
    is_swing_high_4h = pd.Series(False, index=data_4h.index)
    is_swing_low_4h = pd.Series(False, index=data_4h.index)
    
    for i in range(5, len(data_4h) - 5):
        # Swing high: bar at i-2 is highest
        if (high_4h[i-5] < high_4h[i-2] and 
            high_4h[i-3] <= high_4h[i-2] and 
            high_4h[i-2] >= high_4h[i-1] and 
            high_4h[i-2] >= high_4h[i]):
            is_swing_high_4h.iloc[i] = True
        
        # Swing low: bar at i-2 is lowest
        if (low_4h[i-5] > low_4h[i-2] and 
            low_4h[i-3] >= low_4h[i-2] and 
            low_4h[i-2] <= low_4h[i-1] and 
            low_4h[i-2] <= low_4h[i]):
            is_swing_low_4h.iloc[i] = True
    
    # Align swing detection with main data
    data['is_swing_high_4h'] = is_swing_high_4h.reindex(data.index, method='ffill')
    data['is_swing_low_4h'] = is_swing_low_4h.reindex(data.index, method='ffill')
    
    # Track swing points and trend direction
    last_swing_high = np.nan
    last_swing_low = np.nan
    bullish_count = 0
    bearish_count = 0
    trend_direction = "Neutral"
    
    trend_bullish = pd.Series(False, index=data.index)
    trend_bearish = pd.Series(False, index=data.index)
    
    for i in range(len(data)):
        if data['is_swing_high_4h'].iloc[i]:
            last_swing_high = data_4h.loc[data.index[i], 'high'] if data.index[i] in data_4h.index else np.nan
            bullish_count += 1
            bearish_count = 0
        
        if data['is_swing_low_4h'].iloc[i]:
            last_swing_low = data_4h.loc[data.index[i], 'low'] if data.index[i] in data_4h.index else np.nan
            bearish_count += 1
            bullish_count = 0
        
        if bullish_count > 1:
            trend_direction = "Bullish"
            trend_bullish.iloc[i] = True
        elif bearish_count > 1:
            trend_direction = "Bearish"
            trend_bearish.iloc[i] = True
        else:
            trend_direction = "Neutral"
    
    # Entry conditions
    # Based on the strategy context (PDH/PDL sweep with trend confirmation):
    # Long entry: price sweeps PDL (low < prevDayLow) during trading window, trend bullish
    # Short entry: price sweeps PDH (high > prevDayHigh) during trading window, trend bearish
    
    long_entry_cond = (data['previousDayLowTaken']) & in_trading_window & trend_bullish
    short_entry_cond = (data['previousDayHighTaken']) & in_trading_window & trend_bearish
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(data)):
        # Skip if indicators are NaN
        if pd.isna(data['prevDayHigh'].iloc[i]) or pd.isna(data['prevDayLow'].iloc[i]):
            continue
        
        # Check long entry
        if long_entry_cond.iloc[i] and not pd.isna(long_entry_cond.iloc[i]):
            entry_time = data['datetime'].iloc[i]
            entry_ts = int(data['time'].iloc[i])
            entry_price = float(data['close'].iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time.isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        # Check short entry
        if short_entry_cond.iloc[i] and not pd.isna(short_entry_cond.iloc[i]):
            entry_time = data['datetime'].iloc[i]
            entry_ts = int(data['time'].iloc[i])
            entry_price = float(data['close'].iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time.isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries