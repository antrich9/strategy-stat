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
    
    # Default parameters from Pine Script
    lookback = 20
    ret_since = 2
    ret_valid = 2
    trade_direction = 'Both'  # Options: "Long", "Short", "Both"
    rma_length = 14
    
    # Convert price columns to pandas Series
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate pivot highs and lows
    pl = low.rolling(window=lookback + 1).min()
    ph = high.rolling(window=lookback + 1).max()
    
    # Offset pivot values by lookback
    pl_offset = pl.shift(lookback)
    ph_offset = ph.shift(lookback)
    
    # Calculate box boundaries
    s_y_loc = np.where(low.shift(lookback + 1) > low.shift(lookback - 1), 
                       low.shift(lookback - 1), low.shift(lookback + 1))
    r_y_loc = np.where(high.shift(lookback + 1) > high.shift(lookback - 1), 
                       high.shift(lookback + 1), high.shift(lookback - 1))
    
    s_top = pl_offset
    s_bot = pd.Series(s_y_loc, index=df.index)
    r_top = pd.Series(r_y_loc, index=df.index)
    r_bot = ph_offset
    
    # Detect pivot changes
    pl_change = pl != pl.shift(1)
    ph_change = ph != ph.shift(1)
    
    # Box state tracking (simulating var behavior)
    s_break = pd.Series(False, index=df.index)
    r_break = pd.Series(False, index=df.index)
    
    # Repainting mode: simple mode (on)
    r_ton = True
    
    # Breakout detection
    cu = pd.Series(False, index=df.index)  # support break (crossunder)
    co = pd.Series(False, index=df.index)  # resistance break (crossover)
    
    # Iterate through bars to detect breakouts and manage state
    for i in range(lookback + 2, len(df)):
        # Check if pivot changed - reset break flags
        if i > 0 and pl_change.iloc[i]:
            if pd.isna(s_break.iloc[i - 1]) or not s_break.iloc[i - 1]:
                s_break.iloc[i] = False
            else:
                s_break.iloc[i] = False
        elif i > 0:
            s_break.iloc[i] = s_break.iloc[i - 1]
            
        if i > 0 and ph_change.iloc[i]:
            if pd.isna(r_break.iloc[i - 1]) or not r_break.iloc[i - 1]:
                r_break.iloc[i] = False
            else:
                r_break.iloc[i] = False
        elif i > 0:
            r_break.iloc[i] = r_break.iloc[i - 1]
        
        # Check support break (crossunder close below support box bottom)
        s_bot_val = s_bot.iloc[i] if not pd.isna(s_bot.iloc[i]) else np.nan
        if not pd.isna(s_bot_val) and close.iloc[i] < s_bot_val:
            if pd.isna(s_break.iloc[i]) or not s_break.iloc[i]:
                s_break.iloc[i] = True
                cu.iloc[i] = True
        
        # Check resistance break (crossover close above resistance box top)
        r_top_val = r_top.iloc[i] if not pd.isna(r_top.iloc[i]) else np.nan
        if not pd.isna(r_top_val) and close.iloc[i] > r_top_val:
            if pd.isna(r_break.iloc[i]) or not r_break.iloc[i]:
                r_break.iloc[i] = True
                co.iloc[i] = True
    
    # Retest conditions for support
    bars_since_sbreak = pd.Series(0, index=df.index)
    for i in range(lookback + 2, len(df)):
        if s_break.iloc[i]:
            bars_since_sbreak.iloc[i] = 0
        elif i > 0 and bars_since_sbreak.iloc[i - 1] >= 0:
            bars_since_sbreak.iloc[i] = bars_since_sbreak.iloc[i - 1] + 1
    
    s1 = (bars_since_sbreak > ret_since) & (high >= s_top) & (close <= s_bot)
    s2 = (bars_since_sbreak > ret_since) & (high >= s_top) & (close >= s_bot) & (close <= s_top)
    s3 = (bars_since_sbreak > ret_since) & (high >= s_bot) & (high <= s_top)
    s4 = (bars_since_sbreak > ret_since) & (high >= s_bot) & (high <= s_top) & (close < s_bot)
    
    # Retest conditions for resistance
    bars_since_rbreak = pd.Series(0, index=df.index)
    for i in range(lookback + 2, len(df)):
        if r_break.iloc[i]:
            bars_since_rbreak.iloc[i] = 0
        elif i > 0 and bars_since_rbreak.iloc[i - 1] >= 0:
            bars_since_rbreak.iloc[i] = bars_since_rbreak.iloc[i - 1] + 1
    
    r1 = (bars_since_rbreak > ret_since) & (low <= r_bot) & (close >= r_top)
    r2 = (bars_since_rbreak > ret_since) & (low <= r_bot) & (close <= r_top) & (close >= r_bot)
    r3 = (bars_since_rbreak > ret_since) & (low <= r_top) & (low >= r_bot)
    r4 = (bars_since_rbreak > ret_since) & (low <= r_top) & (low >= r_bot) & (close > r_top)
    
    # Combined retest conditions
    s_ret_active = s1 | s2 | s3 | s4
    r_ret_active = r1 | r2 | r3 | r4
    
    # Detect retest events (rising edge)
    s_ret_event = s_ret_active & ~s_ret_active.shift(1).fillna(False)
    r_ret_event = r_ret_active & ~r_ret_active.shift(1).fillna(False)
    
    # Retest valid conditions
    s_ret_valid = pd.Series(False, index=df.index)
    r_ret_valid = pd.Series(False, index=df.index)
    
    for i in range(lookback + 2, len(df)):
        if s_ret_event.iloc[i]:
            bars_since_ret = 0
            ret_occurred = False
            ret_value = high.iloc[i]
        elif i > 0:
            bars_since_ret = bars_since_ret + 1 if 'bars_since_ret' in dir() else 0
            
        if s_ret_active.iloc[i] and (i == 0 or not s_ret_event.iloc[i]):
            ret_valid = (bars_since_ret > 0) & (bars_since_ret <= ret_valid) & (high.iloc[i] >= ret_value if 'ret_value' in dir() else False) & (not ret_occurred if 'ret_occurred' in dir() else False)
            if ret_valid:
                s_ret_valid.iloc[i] = True
                ret_occurred = True
        
        if r_ret_event.iloc[i]:
            bars_since_ret_r = 0
            ret_occurred_r = False
            ret_value_r = low.iloc[i]
        elif i > 0:
            bars_since_ret_r = bars_since_ret_r + 1 if 'bars_since_ret_r' in dir() else 0
            
        if r_ret_active.iloc[i] and (i == 0 or not r_ret_event.iloc[i]):
            ret_valid_r = (bars_since_ret_r > 0) & (bars_since_ret_r <= ret_valid) & (low.iloc[i] <= ret_value_r if 'ret_value_r' in dir() else False) & (not ret_occurred_r if 'ret_occurred_r' in dir() else False)
            if ret_valid_r:
                r_ret_valid.iloc[i] = True
                ret_occurred_r = True
    
    # Build entry signals
    entries = []
    trade_num = 1
    
    # Long entries: resistance breakout with valid retest
    # Short entries: support breakout with valid retest
    
    for i in range(lookback + 3, len(df)):
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = float(close.iloc[i])
        
        # Long entry condition
        if trade_direction in ['Long', 'Both']:
            if co.iloc[i] and r_ret_valid.iloc[i]:
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
        
        # Short entry condition
        if trade_direction in ['Short', 'Both']:
            if cu.iloc[i] and s_ret_valid.iloc[i]:
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
    
    return entries