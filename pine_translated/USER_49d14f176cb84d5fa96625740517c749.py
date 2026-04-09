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
    close = df['close']
    high = df['high']
    low = df['low']
    open_arr = df['open']
    
    n = len(df)
    if n < 5:
        return []
    
    # Parameters
    fib_retracement_level = 0.71
    liquidity_lookback = 20
    fvg_min_size = 0.05
    htf_enabled = True
    fvg_enabled = True
    
    # Wilder ATR(14)
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # Higher timeframe EMA - use 1D request simulation via EWM
    # For true HTF, would need resampled data; here we use EMA on available data
    htf_ema50 = close.ewm(span=50, adjust=False).mean()
    htf_ema200 = close.ewm(span=200, adjust=False).mean()
    htf_trend_bull = htf_ema50 > htf_ema200
    htf_trend_bear = htf_ema50 < htf_ema200
    
    # Initialize state arrays
    recent_swing_high = pd.Series(np.nan, index=df.index)
    recent_swing_low = pd.Series(np.nan, index=df.index)
    
    # Liquidity sweep signals
    liquidity_sweep_bull = pd.Series(False, index=df.index)
    liquidity_sweep_bear = pd.Series(False, index=df.index)
    
    # Fair Value Gap
    bull_fvg = pd.Series(False, index=df.index)
    bear_fvg = pd.Series(False, index=df.index)
    
    # BOS direction tracking
    bos_dir = pd.Series(0, index=df.index)
    prev_bos_dir = 0
    
    # Fibonacci state
    fib_direction_arr = pd.Series(0, index=df.index)
    fib_range_high = pd.Series(np.nan, index=df.index)
    fib_range_low = pd.Series(np.nan, index=df.index)
    fib_71_level = pd.Series(np.nan, index=df.index)
    
    fib_direction = 0
    fib_high = np.nan
    fib_low = np.nan
    fib_71 = np.nan
    
    # Process bars
    for i in range(liquidity_lookback + 2, n):
        if i < 5:
            continue
            
        # Swing high/low via pivothigh/pivotlow (simplified local detection)
        # Check if current bar is a local high/low within lookback
        is_local_high = True
        is_local_low = True
        for j in range(1, liquidity_lookback + 1):
            if i - j >= 0:
                if high.iloc[i] < high.iloc[i - j]:
                    is_local_high = False
                if low.iloc[i] > low.iloc[i - j]:
                    is_local_low = False
            if i + j < n:
                if high.iloc[i] < high.iloc[i + j]:
                    is_local_high = False
                if low.iloc[i] > low.iloc[i + j]:
                    is_local_low = False
        
        # Update recent swing levels
        if is_local_high:
            recent_swing_high.iloc[i] = high.iloc[i]
        else:
            recent_swing_high.iloc[i] = recent_swing_high.iloc[i-1] if not pd.isna(recent_swing_high.iloc[i-1]) else np.nan
            
        if is_local_low:
            recent_swing_low.iloc[i] = low.iloc[i]
        else:
            recent_swing_low.iloc[i] = recent_swing_low.iloc[i-1] if not pd.isna(recent_swing_low.iloc[i-1]) else np.nan
        
        # Recalculate with proper pivot detection
        # pivothigh(high, 20, 20) means look back 20 bars, pivot at 20 bars back from current
        swing_high_val = np.nan
        swing_low_val = np.nan
        
        if i >= liquidity_lookback:
            left = high.iloc[i-liquidity_lookback:i].max()
            if high.iloc[i] == left and high.iloc[i] >= high.iloc[i-1] if i-1 >= 0 else True:
                swing_high_val = high.iloc[i]
        
        if i >= liquidity_lookback:
            left = low.iloc[i-liquidity_lookback:i].max()
            # pivotlow: find lowest low in lookback window
            pass
        
        # Better pivot detection
        if i >= liquidity_lookback:
            window_high = high.iloc[i-liquidity_lookback:i].values
            window_low = low.iloc[i-liquidity_lookback:i].values
            
            max_idx = np.argmax(window_high)
            min_idx = np.argmin(window_low)
            
            if max_idx == len(window_high) - 1:
                swing_high_val = high.iloc[i]
            if min_idx == len(window_low) - 1:
                swing_low_val = low.iloc[i]
        
        if not pd.isna(swing_high_val):
            recent_swing_high.iloc[i] = swing_high_val
        if not pd.isna(swing_low_val):
            recent_swing_low.iloc[i] = swing_low_val
        
        # Liquidity sweep detection
        rsh = recent_swing_high.iloc[i]
        rsl = recent_swing_low.iloc[i]
        
        if not pd.isna(rsl) and low.iloc[i] < rsl and close.iloc[i] > rsl:
            liquidity_sweep_bull.iloc[i] = True
        if not pd.isna(rsh) and high.iloc[i] > rsh and close.iloc[i] < rsh:
            liquidity_sweep_bear.iloc[i] = True
        
        # Fair Value Gap
        if i >= 2:
            bull_fvg.iloc[i] = low.iloc[i] > high.iloc[i-2] and (low.iloc[i] - high.iloc[i-2]) / close.iloc[i] > fvg_min_size / 100
            bear_fvg.iloc[i] = high.iloc[i] < low.iloc[i-2] and (low.iloc[i-2] - high.iloc[i]) / close.iloc[i] > fvg_min_size / 100
        
        # BOS direction
        current_bos_dir = 0
        if liquidity_sweep_bull.iloc[i] and close.iloc[i] > high.iloc[i-1]:
            current_bos_dir = 1
        elif liquidity_sweep_bear.iloc[i] and close.iloc[i] < low.iloc[i-1]:
            current_bos_dir = -1
        
        bos_dir.iloc[i] = current_bos_dir
        
        # Fibonacci tracking - on BOS change
        if current_bos_dir != 0 and current_bos_dir != prev_bos_dir:
            if current_bos_dir == 1:
                fib_direction = 1
                fib_low = recent_swing_low.iloc[i] if not pd.isna(recent_swing_low.iloc[i]) else low.iloc[i]
                fib_high = recent_swing_high.iloc[i] if not pd.isna(recent_swing_high.iloc[i]) else high.iloc[i]
            elif current_bos_dir == -1:
                fib_direction = -1
                fib_high = recent_swing_high.iloc[i] if not pd.isna(recent_swing_high.iloc[i]) else high.iloc[i]
                fib_low = recent_swing_low.iloc[i] if not pd.isna(recent_swing_low.iloc[i]) else low.iloc[i]
        
        # Update Fibonacci range dynamically while in active direction
        if fib_direction == 1:
            if not pd.isna(fib_high):
                fib_high = max(fib_high, high.iloc[i])
            else:
                fib_high = high.iloc[i]
            if not pd.isna(fib_low):
                fib_71 = fib_high - (fib_high - fib_low) * fib_retracement_level
        elif fib_direction == -1:
            if not pd.isna(fib_low):
                fib_low = min(fib_low, low.iloc[i])
            else:
                fib_low = low.iloc[i]
            if not pd.isna(fib_high):
                fib_71 = fib_low + (fib_high - fib_low) * fib_retracement_level
        
        fib_direction_arr.iloc[i] = fib_direction
        fib_range_high.iloc[i] = fib_high
        fib_range_low.iloc[i] = fib_low
        fib_71_level.iloc[i] = fib_71
        
        prev_bos_dir = current_bos_dir
    
    # Build entry conditions
    bull_htf_ok = (~htf_enabled) | htf_trend_bull
    bull_liquidity_sweep = liquidity_sweep_bull
    bull_bos = (bos_dir == 1)
    bull_fvg_present = (~fvg_enabled) | bull_fvg.shift(1).fillna(False) | bull_fvg.shift(2).fillna(False) | bull_fvg.shift(3).fillna(False)
    bull_at_fib = (~pd.isna(fib_71_level)) & (close <= fib_71_level) & (close >= fib_71_level * 0.99)
    
    bullish_entry = bull_htf_ok & bull_liquidity_sweep & bull_bos & bull_fvg_present & bull_at_fib
    
    bear_htf_ok = (~htf_enabled) | htf_trend_bear
    bear_liquidity_sweep = liquidity_sweep_bear
    bear_bos = (bos_dir == -1)
    bear_fvg_present = (~fvg_enabled) | bear_fvg.shift(1).fillna(False) | bear_fvg.shift(2).fillna(False) | bear_fvg.shift(3).fillna(False)
    bear_at_fib = (~pd.isna(fib_71_level)) & (close >= fib_71_level) & (close <= fib_71_level * 1.01)
    
    bearish_entry = bear_htf_ok & bear_liquidity_sweep & bear_bos & bear_fvg_present & bear_at_fib
    
    entries = []
    trade_num = 1
    
    for i in range(liquidity_lookback + 2, n):
        if pd.isna(atr.iloc[i]):
            continue
        
        entry_price = close.iloc[i]
        
        if bullish_entry.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000 if entry_ts > 1e10 else entry_ts, tz=timezone.utc).isoformat()
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
        
        if bearish_entry.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000 if entry_ts > 1e10 else entry_ts, tz=timezone.utc).isoformat()
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