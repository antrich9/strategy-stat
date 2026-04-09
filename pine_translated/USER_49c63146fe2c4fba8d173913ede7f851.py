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
    
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Detect NY hour for each bar
    df['ny_hour'] = df['time'].dt.tz_convert('America/New_York').dt.hour
    
    # Asian session: NY hour >= 19
    in_asian_session = df['ny_hour'] >= 19
    
    # Session state tracking
    asian_high = pd.Series([np.nan] * len(df), dtype=float)
    asian_low = pd.Series([np.nan] * len(df), dtype=float)
    temp_asian_high = pd.Series([np.nan] * len(df), dtype=float)
    temp_asian_low = pd.Series([np.nan] * len(df), dtype=float)
    
    was_in_session = False
    asian_session_ended = False
    
    # Previous day high/low tracking
    pd_high = pd.Series([np.nan] * len(df), dtype=float)
    pd_low = pd.Series([np.nan] * len(df), dtype=float)
    temp_pd_high = pd.Series([np.nan] * len(df), dtype=float)
    temp_pd_low = pd.Series([np.nan] * len(df), dtype=float)
    
    # Sweep tracking flags
    asian_swept_high = False
    asian_swept_low = False
    pd_swept_high = False
    pd_swept_low = False
    
    # Daily bias from security
    c1_high = pd.Series([np.nan] * len(df), dtype=float)
    c1_low = pd.Series([np.nan] * len(df), dtype=float)
    c2_close = pd.Series([np.nan] * len(df), dtype=float)
    c2_high = pd.Series([np.nan] * len(df), dtype=float)
    c2_low = pd.Series([np.nan] * len(df), dtype=float)
    
    # Daily bias calculation
    bullish_bias = pd.Series([False] * len(df), dtype=bool)
    bearish_bias = pd.Series([False] * len(df), dtype=bool)
    
    # Process each bar
    for i in range(len(df)):
        current_time = df['time'].iloc[i]
        ny_hour = df['ny_hour'].iloc[i]
        in_asian = in_asian_session.iloc[i]
        high_val = df['high'].iloc[i]
        low_val = df['low'].iloc[i]
        close_val = df['close'].iloc[i]
        
        # Session just started detection
        session_just_started = in_asian and not was_in_session
        
        if session_just_started:
            temp_asian_high.iloc[i] = high_val
            temp_asian_low.iloc[i] = low_val
        elif in_asian:
            if pd.notna(temp_asian_high.iloc[i-1]):
                temp_asian_high.iloc[i] = max(temp_asian_high.iloc[i-1], high_val)
            else:
                temp_asian_high.iloc[i] = high_val
            if pd.notna(temp_asian_low.iloc[i-1]):
                temp_asian_low.iloc[i] = min(temp_asian_low.iloc[i-1], low_val)
            else:
                temp_asian_low.iloc[i] = low_val
        else:
            temp_asian_high.iloc[i] = temp_asian_high.iloc[i-1] if pd.notna(temp_asian_high.iloc[i-1]) else np.nan
            temp_asian_low.iloc[i] = temp_asian_low.iloc[i-1] if pd.notna(temp_asian_low.iloc[i-1]) else np.nan
        
        # Asian session ended detection (was in session previous bar, not now)
        if i > 0:
            asian_session_ended = was_in_session and not in_asian
        
        if asian_session_ended:
            asian_high.iloc[i] = temp_asian_high.iloc[i-1]
            asian_low.iloc[i] = temp_asian_low.iloc[i-1]
            asian_swept_high = False
            asian_swept_low = False
        
        # Asian sweep detection
        if not asian_swept_high and pd.notna(asian_high.iloc[i]) and high_val > asian_high.iloc[i]:
            asian_swept_high = True
        if not asian_swept_low and pd.notna(asian_low.iloc[i]) and low_val < asian_low.iloc[i]:
            asian_swept_low = True
        
        # Previous day high/low detection
        current_day = current_time.date()
        prev_day_from_prev = current_day if i == 0 else df['time'].iloc[i-1].date()
        new_day = current_day != prev_day_from_prev
        
        if new_day and i > 0:
            pd_high.iloc[i] = temp_pd_high.iloc[i-1]
            pd_low.iloc[i] = temp_pd_low.iloc[i-1]
            pd_swept_high = False
            pd_swept_low = False
        else:
            pd_high.iloc[i] = pd_high.iloc[i-1] if i > 0 and pd.notna(pd_high.iloc[i-1]) else np.nan
            pd_low.iloc[i] = pd_low.iloc[i-1] if i > 0 and pd.notna(pd_low.iloc[i-1]) else np.nan
        
        if in_asian:
            if pd.isna(temp_pd_high.iloc[i]):
                temp_pd_high.iloc[i] = high_val
            else:
                temp_pd_high.iloc[i] = max(temp_pd_high.iloc[i-1] if pd.notna(temp_pd_high.iloc[i-1]) else high_val, high_val)
            if pd.isna(temp_pd_low.iloc[i]):
                temp_pd_low.iloc[i] = low_val
            else:
                temp_pd_low.iloc[i] = min(temp_pd_low.iloc[i-1] if pd.notna(temp_pd_low.iloc[i-1]) else low_val, low_val)
        else:
            temp_pd_high.iloc[i] = temp_pd_high.iloc[i-1] if pd.notna(temp_pd_high.iloc[i-1]) else np.nan
            temp_pd_low.iloc[i] = temp_pd_low.iloc[i-1] if pd.notna(temp_pd_low.iloc[i-1]) else np.nan
        
        # PD sweep detection
        pd_sweep_high_now = not pd_swept_high and pd.notna(pd_high.iloc[i]) and high_val > pd_high.iloc[i]
        pd_sweep_low_now = not pd_swept_low and pd.notna(pd_low.iloc[i]) and low_val < pd_low.iloc[i]
        
        if pd_sweep_high_now:
            pd_swept_high = True
        if pd_sweep_low_now:
            pd_swept_low = True
        
        # Daily bias calculation using shifted bars
        if i >= 3:
            # c1 is bar[2], c2 is bar[1] in Pine notation
            c1_h = df['high'].iloc[i-2]
            c1_l = df['low'].iloc[i-2]
            c2_c = df['close'].iloc[i-1]
            c2_h = df['high'].iloc[i-1]
            c2_l = df['low'].iloc[i-1]
            
            bullish_bias.iloc[i] = (c2_c > c1_h) or (c2_l < c1_l and c2_c > c1_l)
            bearish_bias.iloc[i] = (c2_c < c1_l) or (c2_h > c1_h and c2_c < c1_h)
        
        was_in_session = in_asian
    
    # Individual directional sweep conditions
    asian_long_ok = asian_swept_low and not asian_swept_high
    asian_short_ok = asian_swept_high and not asian_swept_low
    pd_long_ok = pd_swept_low and not pd_swept_high
    pd_short_ok = pd_swept_high and not pd_swept_low
    bias_long_ok = bullish_bias
    bias_short_ok = bearish_bias
    
    # Combined sweep conditions (All Three mode - default)
    long_sweep_ok = asian_long_ok & pd_long_ok & bias_long_ok
    short_sweep_ok = asian_short_ok & pd_short_ok & bias_short_ok
    
    # Build entry conditions and iterate
    entries = []
    trade_num = 1
    
    # Convert time back to unix timestamp for output
    df['ts'] = (df['time'].astype('int64') // 10**9)
    
    for i in range(len(df)):
        if pd.isna(long_sweep_ok.iloc[i]) and pd.isna(short_sweep_ok.iloc[i]):
            continue
        
        direction = None
        if long_sweep_ok.iloc[i]:
            direction = 'long'
        elif short_sweep_ok.iloc[i]:
            direction = 'short'
        
        if direction is None:
            continue
        
        entry_ts = int(df['ts'].iloc[i])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        entry_price = df['close'].iloc[i]
        
        entries.append({
            'trade_num': trade_num,
            'direction': direction,
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