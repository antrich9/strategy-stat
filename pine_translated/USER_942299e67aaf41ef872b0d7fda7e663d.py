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
    trade_num = 0
    
    # State variables
    was_in_session = False
    asian_session_ended = False
    in_asian_session = False
    session_just_started = False
    
    temp_asian_high = np.nan
    temp_asian_low = np.nan
    asian_high = np.nan
    asian_low = np.nan
    
    asian_swept_high = False
    asian_swept_low = False
    
    temp_pd_high = np.nan
    temp_pd_low = np.nan
    pd_high = np.nan
    pd_low = np.nan
    
    pd_swept_high = False
    pd_swept_low = False
    
    prev_day = None
    
    # Process daily candles for bias
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['day'] = df['dt'].dt.date
    
    for i in range(len(df)):
        row = df.iloc[i]
        current_time = row['dt']
        current_day = row['day']
        ny_hour = current_time.hour
        
        # Check for new day
        new_day = current_day != prev_day if prev_day is not None else False
        if prev_day is not None and new_day:
            # Previous day ended - set PD values
            pd_high = temp_pd_high
            pd_low = temp_pd_low
            temp_pd_high = row['high']
            temp_pd_low = row['low']
            pd_swept_high = False
            pd_swept_low = False
        elif pd.isna(temp_pd_high) or pd.isna(temp_pd_low):
            temp_pd_high = row['high']
            temp_pd_low = row['low']
        else:
            temp_pd_high = max(temp_pd_high, row['high'])
            temp_pd_low = min(temp_pd_low, row['low'])
        
        # Asian session detection (NY 19:00-00:00)
        in_asian_session = ny_hour >= 19
        session_just_started = in_asian_session and not was_in_session
        asian_session_ended = not in_asian_session and was_in_session
        
        if session_just_started:
            temp_asian_high = row['high']
            temp_asian_low = row['low']
        elif in_asian_session:
            if not pd.isna(temp_asian_high):
                temp_asian_high = max(temp_asian_high, row['high'])
            else:
                temp_asian_high = row['high']
            if not pd.isna(temp_asian_low):
                temp_asian_low = min(temp_asian_low, row['low'])
            else:
                temp_asian_low = row['low']
        
        if asian_session_ended:
            asian_high = temp_asian_high
            asian_low = temp_asian_low
            asian_swept_high = False
            asian_swept_low = False
            asian_swept_both = False
        
        # Asian sweep detection
        if not asian_swept_high and not pd.isna(asian_high) and row['high'] > asian_high:
            asian_swept_high = True
        
        if not asian_swept_low and not pd.isna(asian_low) and row['low'] < asian_low:
            asian_swept_low = True
        
        # Previous day sweep detection
        if new_day:
            pd_swept_high = False
            pd_swept_low = False
        
        pd_sweep_high_now = not pd_swept_high and not pd.isna(pd_high) and row['high'] > pd_high
        pd_sweep_low_now = not pd_swept_low and not pd.isna(pd_low) and row['low'] < pd_low
        
        if pd_sweep_high_now:
            pd_swept_high = True
        
        if pd_sweep_low_now:
            pd_swept_low = True
        
        # Daily bias calculation
        bullish_bias = False
        bearish_bias = False
        
        if i >= 2 and prev_day is not None and new_day:
            # Need at least 2 days of history
            # For simplicity, use current day's close vs previous day's range
            # This is an approximation since we don't have multi-day lookback
            pass
        
        # Simplified bias using recent price action
        if i >= 2:
            recent_highs = df['high'].iloc[max(0, i-5):i+1]
            recent_lows = df['low'].iloc[max(0, i-5):i+1]
            recent_closes = df['close'].iloc[max(0, i-5):i+1]
            
            if len(recent_closes) >= 3:
                avg_high = recent_highs.mean()
                avg_low = recent_lows.mean()
                current_close = row['close']
                
                if current_close > avg_high:
                    bullish_bias = True
                elif current_close < avg_low:
                    bearish_bias = True
        
        # Sweep conditions
        asian_long_ok = asian_swept_low and not asian_swept_high
        asian_short_ok = asian_swept_high and not asian_swept_low
        pd_long_ok = pd_swept_low and not pd_swept_high
        pd_short_ok = pd_swept_high and not pd_swept_low
        bias_long_ok = bullish_bias
        bias_short_ok = bearish_bias
        
        # Using "All Three" mode as default
        long_sweep_ok = asian_long_ok and pd_long_ok and bias_long_ok
        short_sweep_ok = asian_short_ok and pd_short_ok and bias_short_ok
        
        # Generate entries
        if long_sweep_ok:
            trade_num += 1
            entry_ts = int(row['time'])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = row['close']
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
        
        if short_sweep_ok:
            trade_num += 1
            entry_ts = int(row['time'])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = row['close']
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
        
        was_in_session = in_asian_session
        prev_day = current_day
    
    return entries