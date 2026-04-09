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
    
    fvgTH = 0.5
    atr_period = 144
    london_tz_offset_morning_start = 6 * 3600 + 45 * 60
    london_tz_offset_morning_end = 9 * 3600 + 45 * 60
    london_tz_offset_afternoon_start = 14 * 3600 + 45 * 60
    london_tz_offset_afternoon_end = 16 * 3600 + 45 * 60
    
    df = df.copy()
    df['morning_window'] = False
    df['afternoon_window'] = False
    
    for i in range(len(df)):
        ts = df['time'].iloc[i]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        seconds_since_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second
        
        if seconds_since_midnight >= london_tz_offset_morning_start and seconds_since_midnight < london_tz_offset_morning_end:
            df.iloc[i, df.columns.get_loc('morning_window')] = True
        elif seconds_since_midnight >= london_tz_offset_afternoon_start and seconds_since_midnight < london_tz_offset_afternoon_end:
            df.iloc[i, df.columns.get_loc('afternoon_window')] = True
    
    df['is_within_time_window'] = df['morning_window'] | df['afternoon_window']
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    prev_day_high = high.shift(1).rolling(window=144, min_periods=144).max().shift(1)
    prev_day_low = low.shift(1).rolling(window=144, min_periods=144).min().shift(1)
    df['prev_day_high'] = prev_day_high
    df['prev_day_low'] = prev_day_low
    
    df['previous_day_high_taken'] = high > prev_day_high
    df['previous_day_low_taken'] = low < prev_day_low
    
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_values = pd.Series(index=true_range.index, dtype=float)
    atr_values.iloc[0] = true_range.iloc[0]
    for i in range(1, len(true_range)):
        atr_values.iloc[i] = (atr_values.iloc[i-1] * (atr_period - 1) + true_range.iloc[i]) / atr_period
    
    df['atr'] = atr_values
    df['atr_filtered'] = df['atr'] * fvgTH
    
    df['bullG'] = low > high.shift(1)
    df['bearG'] = high < low.shift(1)
    
    df['bull_fvg'] = (
        (low - high.shift(2)) > df['atr_filtered'] &
        low > high.shift(2) &
        close.shift(1) > high.shift(2) &
        ~(df['bullG'] | df['bullG'].shift(1))
    )
    
    df['bear_fvg'] = (
        (low.shift(2) - high) > df['atr_filtered'] &
        high < low.shift(2) &
        close.shift(1) < low.shift(2) &
        ~(df['bearG'] | df['bearG'].shift(1))
    )
    
    df['long_entry'] = df['is_within_time_window'] & df['bull_fvg']
    df['short_entry'] = df['is_within_time_window'] & df['bear_fvg']
    
    entries = []
    trade_num = 1
    
    for i in range(2, len(df)):
        if df['atr'].iloc[i] != df['atr'].iloc[i] or np.isnan(df['atr'].iloc[i]):
            continue
        
        if df['long_entry'].iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        elif df['short_entry'].iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries