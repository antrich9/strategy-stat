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
    
    # Extract time components for timezone-aware filtering
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['dayofweek'] = df['datetime'].dt.dayofweek  # 0=Monday, 4=Friday
    
    # Filter for London trading hours (weekdays only)
    # Window 1: 07:45-11:45 London time
    # Window 2: 14:00-14:45 London time
    def in_trading_window(row):
        hour = row['hour']
        minute = row['minute']
        dow = row['dayofweek']
        if dow >= 5:  # Saturday=5, Sunday=6
            return False
        # Window 1: 07:45-11:45
        if (hour == 7 and minute >= 45) or (8 <= hour <= 10) or (hour == 11 and minute <= 45):
            return True
        # Window 2: 14:00-14:45
        if (hour == 14 and minute <= 45):
            return True
        return False
    
    df['in_window'] = df.apply(in_trading_window, axis=1)
    
    # Calculate swing highs/lows using daily data
    # Simulate daily high/low using daily resampling
    df_daily = df.copy()
    df_daily.set_index('datetime', inplace=True)
    daily_ohlc = df_daily.resample('D').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
    
    # Get previous day's high/low
    daily_ohlc['prev_high'] = daily_ohlc['high'].shift(1)
    daily_ohlc['prev_low'] = daily_ohlc['low'].shift(1)
    daily_ohlc['high_2'] = daily_ohlc['high'].shift(2)
    daily_ohlc['low_2'] = daily_ohlc['low'].shift(2)
    daily_ohlc['high_3'] = daily_ohlc['high'].shift(3)
    daily_ohlc['low_3'] = daily_ohlc['low'].shift(3)
    daily_ohlc['high_4'] = daily_ohlc['high'].shift(4)
    daily_ohlc['low_4'] = daily_ohlc['low'].shift(4)
    
    # Swing detection: dailyHigh21 < dailyHigh22 and dailyHigh11[3] < dailyHigh22 and dailyHigh11[4] < dailyHigh22
    # dailyHigh21 = prev_day_high, dailyHigh22 = 2 days ago high
    is_swing_high = (daily_ohlc['prev_high'] < daily_ohlc['high_2']) & \
                    (daily_ohlc['high_3'] < daily_ohlc['high_2']) & \
                    (daily_ohlc['high_4'] < daily_ohlc['high_2'])
    is_swing_low = (daily_ohlc['prev_low'] > daily_ohlc['low_2']) & \
                   (daily_ohlc['low_3'] > daily_ohlc['low_2']) & \
                   (daily_ohlc['low_4'] > daily_ohlc['low_2'])
    
    # Track last swing type
    last_swing_type = "none"
    last_swing_high = np.nan
    last_swing_low = np.nan
    
    # Sweep detection variables
    swept_high = False
    swept_low = False
    prev_day_high = np.nan
    prev_day_low = np.nan
    
    # For 4H FVG detection, we need to resample to 4H
    df_4h = df.copy()
    df_4h.set_index('datetime', inplace=True)
    ohlc_4h = df_4h.resample('4H').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'open': 'first',
        'volume': 'sum'
    }).dropna()
    
    # Shift for FVG detection
    ohlc_4h['high_1'] = ohlc_4h['high'].shift(1)
    ohlc_4h['low_1'] = ohlc_4h['low'].shift(1)
    ohlc_4h['high_2'] = ohlc_4h['high'].shift(2)
    ohlc_4h['low_2'] = ohlc_4h['low'].shift(2)
    ohlc_4h['close_1'] = ohlc_4h['close'].shift(1)
    ohlc_4h['close_2'] = ohlc_4h['close'].shift(2)
    ohlc_4h['volume_1'] = ohlc_4h['volume'].shift(1)
    
    # FVG conditions: bfvg = low_4h > high_4h_2, sfvg = high_4h < low_4h_2
    ohlc_4h['bfvg'] = ohlc_4h['low'] > ohlc_4h['high_2']
    ohlc_4h['sfvg'] = ohlc_4h['high'] < ohlc_4h['low_2']
    
    # Track first bar of day
    df['date'] = df['datetime'].dt.date
    df['is_new_day'] = df['date'] != df['date'].shift(1)
    df['is_new_day'].iloc[0] = True
    
    # Map daily swing info to main df
    df['date_only'] = df['datetime'].dt.normalize()
    swing_high_series = pd.Series(index=daily_ohlc.index, dtype=float)
    swing_low_series = pd.Series(index=daily_ohlc.index, dtype=float)
    swing_high_series[daily_ohlc.index] = daily_ohlc['high_2'].where(is_swing_high, np.nan)
    swing_low_series[daily_ohlc.index] = daily_ohlc['low_2'].where(is_swing_low, np.nan)
    swing_high_series = swing_high_series.ffill()
    swing_low_series = swing_low_series.ffill()
    
    df['swing_high_2'] = df['date_only'].map(swing_high_series)
    df['swing_low_2'] = df['date_only'].map(swing_low_series)
    
    # Track last swing type in df
    df['swing_high_detected'] = False
    df['swing_low_detected'] = False
    
    # Sweep tracking
    df['swept_high'] = False
    df['swept_low'] = False
    df['prev_day_high'] = np.nan
    df['prev_day_low'] = np.nan
    
    # Map 4H data back to main df
    ohlc_4h_reset = ohlc_4h.reset_index()
    ohlc_4h_reset['time'] = ohlc_4h_reset['datetime']
    
    # Merge 4H FVG signals
    df_merged = df.merge(ohlc_4h_reset[['time', 'bfvg', 'sfvg']], on='time', how='left')
    df['bfvg'] = df_merged['bfvg'].fillna(False)
    df['sfvg'] = df_merged['sfvg'].fillna(False)
    
    # Iterate through bars
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Update sweep state on new day
        if row['is_new_day']:
            # Check if previous day's high was swept
            if i > 0 and not pd.isna(df.iloc[i-1]['prev_day_high']):
                prev_day_high = df.iloc[i-1]['prev_day_high']
                prev_day_low = df.iloc[i-1]['prev_day_low']
                swept_high = False
                swept_low = False
            else:
                prev_day_high = np.nan
                prev_day_low = np.nan
            df.at[df.index[i], 'prev_day_high'] = prev_day_high
            df.at[df.index[i], 'prev_day_low'] = prev_day_low
        else:
            # Track running high/low for current day
            if pd.isna(prev_day_high):
                prev_day_high = row['high']
            else:
                prev_day_high = max(prev_day_high, row['high'])
            if pd.isna(prev_day_low):
                prev_day_low = row['low']
            else:
                prev_day_low = min(prev_day_low, row['low'])
            df.at[df.index[i], 'prev_day_high'] = prev_day_high
            df.at[df.index[i], 'prev_day_low'] = prev_day_low
        
        # Update sweep flags
        sweep_high_now = not swept_high and row['high'] > prev_day_high
        sweep_low_now = not swept_low and row['low'] < prev_day_low
        
        if sweep_high_now:
            swept_high = True
        if sweep_low_now:
            swept_low = True
        
        df.at[df.index[i], 'swept_high'] = swept_high
        df.at[df.index[i], 'swept_low'] = swept_low
        
        # Detect swing
        if row['bfvg'] and row['in_window'] and last_swing_type == "dailyLow":
            # Bullish entry
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(row['time']),
                'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                'entry_price_guess': row['close'],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': row['close'],
                'raw_price_b': row['close']
            }
            entries.append(entry)
            trade_num += 1
            last_swing_type = "none"
        elif row['sfvg'] and row['in_window'] and last_swing_type == "dailyHigh":
            # Bearish entry
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(row['time']),
                'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                'entry_price_guess': row['close'],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': row['close'],
                'raw_price_b': row['close']
            }
            entries.append(entry)
            trade_num += 1
            last_swing_type = "none"
        
        # Update last swing type
        if row['bfvg'] and not pd.isna(row.get('swing_high_2')):
            last_swing_type = "dailyHigh"
        elif row['sfvg'] and not pd.isna(row.get('swing_low_2')):
            last_swing_type = "dailyLow"
    
    return entries