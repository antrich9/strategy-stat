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
    # Parameters
    lookback = 20
    ret_since = 2
    ret_valid_limiter = 2
    
    n = len(df)
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Find pivot points
    pl = pd.Series(np.nan, index=df.index)
    ph = pd.Series(np.nan, index=df.index)
    
    for i in range(lookback, n - lookback):
        p_pos = i - lookback
        # pivot low
        pivot_val = low.iloc[p_pos]
        is_pivot_low = True
        for j in range(p_pos - lookback, p_pos + lookback + 1):
            if j != p_pos and low.iloc[j] <= pivot_val:
                is_pivot_low = False
                break
        if is_pivot_low:
            pl.iloc[i] = pivot_val
        
        # pivot high
        pivot_val = high.iloc[p_pos]
        is_pivot_high = True
        for j in range(p_pos - lookback, p_pos + lookback + 1):
            if j != p_pos and high.iloc[j] >= pivot_val:
                is_pivot_high = False
                break
        if is_pivot_high:
            ph.iloc[i] = pivot_val
    
    # Forward fill pivot values
    pl = pl.ffill()
    ph = ph.ffill()
    
    # Calculate box tops and bottoms
    s_top = pd.Series(np.nan, index=df.index)
    s_bot = pd.Series(np.nan, index=df.index)
    r_top = pd.Series(np.nan, index=df.index)
    r_bot = pd.Series(np.nan, index=df.index)
    
    valid_mask = pl.notna()
    valid_indices = df.index[valid_mask]
    
    for i in valid_indices:
        idx = i - lookback
        if idx >= 1 and idx + 1 < n:
            s_top.iloc[i] = pl.iloc[i]
            s_bot.iloc[i] = low.iloc[idx - 1] if low.iloc[idx + 1] > low.iloc[idx - 1] else low.iloc[idx + 1]
    
    valid_mask = ph.notna()
    valid_indices = df.index[valid_mask]
    
    for i in valid_indices:
        idx = i - lookback
        if idx >= 1 and idx + 1 < n:
            r_bot.iloc[i] = ph.iloc[i]
            r_top.iloc[i] = high.iloc[idx + 1] if high.iloc[idx + 1] > high.iloc[idx - 1] else high.iloc[idx - 1]
    
    # Forward fill box levels
    s_top = s_top.ffill()
    s_bot = s_bot.ffill()
    r_top = r_top.ffill()
    r_bot = r_bot.ffill()
    
    # Calculate breakout conditions (repaint = 'On' mode)
    cu = (close < s_bot) & (close.shift(1) >= s_bot.shift(1))
    co = (close > r_top) & (close.shift(1) <= r_top.shift(1))
    
    # Calculate retest conditions
    sBreak_arr = pd.Series(False, index=df.index)
    rBreak_arr = pd.Series(False, index=df.index)
    
    for i in range(1, n):
        if cu.iloc[i] and not sBreak_arr.iloc[i-1]:
            sBreak_arr.iloc[i] = True
        elif pd.notna(pl.iloc[i]):
            sBreak_arr.iloc[i] = False
        else:
            sBreak_arr.iloc[i] = sBreak_arr.iloc[i-1]
        
        if co.iloc[i] and not rBreak_arr.iloc[i-1]:
            rBreak_arr.iloc[i] = True
        elif pd.notna(ph.iloc[i]):
            rBreak_arr.iloc[i] = False
        else:
            rBreak_arr.iloc[i] = rBreak_arr.iloc[i-1]
    
    # Bars since breakout
    bars_since_sBreak = pd.Series(np.nan, index=df.index)
    bars_since_rBreak = pd.Series(np.nan, index=df.index)
    
    temp_count = 0
    for i in range(n):
        if sBreak_arr.iloc[i]:
            temp_count = 0
        else:
            temp_count += 1
        bars_since_sBreak.iloc[i] = temp_count
    
    temp_count = 0
    for i in range(n):
        if rBreak_arr.iloc[i]:
            temp_count = 0
        else:
            temp_count += 1
        bars_since_rBreak.iloc[i] = temp_count
    
    # Retest conditions for support
    s1 = (bars_since_sBreak > ret_since) & (high >= s_top) & (close <= s_bot)
    s2 = (bars_since_sBreak > ret_since) & (high >= s_top) & (close >= s_bot) & (close <= s_top)
    s3 = (bars_since_sBreak > ret_since) & (high >= s_bot) & (high <= s_top)
    s4 = (bars_since_sBreak > ret_since) & (high >= s_bot) & (high <= s_top) & (close < s_bot)
    
    # Retest conditions for resistance
    r1 = (bars_since_rBreak > ret_since) & (low <= r_bot) & (close >= r_top)
    r2 = (bars_since_rBreak > ret_since) & (low <= r_bot) & (close <= r_top) & (close >= r_bot)
    r3 = (bars_since_rBreak > ret_since) & (low <= r_top) & (low >= r_bot)
    r4 = (bars_since_rBreak > ret_since) & (low <= r_top) & (low >= r_bot) & (close > r_top)
    
    s_ret_active = s1 | s2 | s3 | s4
    r_ret_active = r1 | r2 | r3 | r4
    
    # Calculate retest valid (simplified - ret event when active goes from False to True)
    s_ret_event = s_ret_active & ~s_ret_active.shift(1).fillna(False)
    r_ret_event = r_ret_active & ~r_ret_active.shift(1).fillna(False)
    
    # Ret valid: within bars since event <= ret_valid_limiter
    bars_since_s_ret_event = pd.Series(np.nan, index=df.index)
    bars_since_r_ret_event = pd.Series(np.nan, index=df.index)
    
    temp_count = np.nan
    for i in range(n):
        if s_ret_event.iloc[i]:
            temp_count = 0
        elif not pd.isna(temp_count):
            temp_count += 1
        bars_since_s_ret_event.iloc[i] = temp_count
    
    temp_count = np.nan
    for i in range(n):
        if r_ret_event.iloc[i]:
            temp_count = 0
        elif not pd.isna(temp_count):
            temp_count += 1
        bars_since_r_ret_event.iloc[i] = temp_count
    
    s_ret_valid = (bars_since_s_ret_event > 0) & (bars_since_s_ret_event <= ret_valid_limiter) & s_ret_active
    r_ret_valid = (bars_since_r_ret_event > 0) & (bars_since_r_ret_event <= ret_valid_limiter) & r_ret_active
    
    # Long entry conditions
    long_cond = co | r_ret_valid
    
    # Short entry conditions
    short_cond = cu | s_ret_valid
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(1, n):
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = close.iloc[i]
        
        if long_cond.iloc[i]:
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
        
        if short_cond.iloc[i]:
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