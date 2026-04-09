import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    Convert Pine Script entry logic to Python.
    """
    entries = []
    trade_num = 1
    
    # Ensure required columns exist
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        return entries
    
    # Create a copy to avoid modifying original
    data = df.copy()
    
    # Calculate previous day high/low (using daily aggregation)
    data['day'] = pd.to_datetime(data['time'], unit='s', utc=True).dt.date
    daily_agg = data.groupby('day').agg({
        'high': 'max',
        'low': 'min'
    }).reset_index()
    daily_agg.columns = ['day', 'day_high', 'day_low']
    
    # Shift to get previous day values
    daily_agg['prev_day_high'] = daily_agg['day_high'].shift(1)
    daily_agg['prev_day_low'] = daily_agg['day_low'].shift(1)
    
    # Merge back
    data = data.merge(daily_agg[['day', 'prev_day_high', 'prev_day_low']], on='day', how='left')
    
    # Fill forward the previous day values
    data['prev_day_high'] = data['prev_day_high'].ffill()
    data['prev_day_low'] = data['prev_day_low'].ffill()
    
    # Detect sweeps
    data['sweep_pdh'] = data['close'] > data['prev_day_high']
    data['sweep_pdl'] = data['close'] < data['prev_day_low']
    
    # Track flag states (carry forward once triggered)
    flag_pdh = False
    flag_pdl = False
    
    # Time filtering
    data['time_dt'] = pd.to_datetime(data['time'], unit='s', utc=True)
    data['hour'] = data['time_dt'].dt.hour + data['time_dt'].dt.minute / 60.0
    
    # 0700-0959 and 1200-1459 sessions
    data['in_session1'] = (data['hour'] >= 7.0) & (data['hour'] < 10.0)
    data['in_session2'] = (data['hour'] >= 12.0) & (data['hour'] < 15.0)
    data['in_session'] = data['in_session1'] | data['in_session2']
    
    # OB detection
    data['is_up'] = data['close'] > data['open']
    data['is_down'] = data['close'] < data['open']
    
    # OB Up: down candle then up candle where close > previous high
    data['ob_up'] = (data['is_down'].shift(1)) & (data['is_up']) & (data['close'] > data['high'].shift(1))
    
    # OB Down: up candle then down candle where close < previous low
    data['ob_down'] = (data['is_up'].shift(1)) & (data['is_down']) & (data['close'] < data['low'].shift(1))
    
    # FVG detection
    data['fvg_up'] = data['low'] > data['high'].shift(2)
    data['fvg_down'] = data['high'] < data['low'].shift(2)
    
    # Stacked OB+FVG signals
    data['bullish_stack'] = data['ob_up'] & data['fvg_up'].shift(1)
    data['bearish_stack'] = data['ob_down'] & data['fvg_down'].shift(1)
    
    # Reset flags on new day
    data['is_new_day'] = data['day'] != data['day'].shift(1)
    
    # Iterate through bars
    prev_day = None
    for i in range(len(data)):
        row = data.iloc[i]
        current_day = row['day']
        
        # Reset flags on new day
        if pd.notna(prev_day) and current_day != prev_day:
            flag_pdh = False
            flag_pdl = False
        
        prev_day = current_day
        
        # Update flags on sweep
        if row['sweep_pdh'] and pd.notna(row['prev_day_high']):
            flag_pdh = True
        
        if row['sweep_pdl'] and pd.notna(row['prev_day_low']):
            flag_pdl = True
        
        # Check for valid indicators
        if pd.isna(row['prev_day_high']) or pd.isna(row['prev_day_low']):
            continue
        
        if not row['in_session']:
            continue
        
        # Entry conditions based on sweep flags and stacked patterns
        entry_price = row['close']
        
        # Long entry: previous day high swept + bullish stack
        if flag_pdh and row['bullish_stack']:
            ts = int(row['time'])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            # Reset flag after entry
            flag_pdh = False
        
        # Short entry: previous day low swept + bearish stack
        if flag_pdl and row['bearish_stack']:
            ts = int(row['time'])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            # Reset flag after entry
            flag_pdl = False
    
    return entries