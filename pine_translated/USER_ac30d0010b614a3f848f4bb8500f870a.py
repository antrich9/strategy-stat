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
    
    # Parameters from Pine Script inputs
    lookback = 200
    atr_length = 14
    fvg_wait_bars = 10
    fvg_min_ticks = 3
    mintick = 0.01  # Assuming mintick for calculation (placeholder)
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate True Range and ATR (Wilder method)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/atr_length, adjust=False).mean()
    
    # Range high/low (highest/lowest over lookback)
    range_high = high.rolling(lookback).max()
    range_low = low.rolling(lookback).min()
    
    # Tolerance for sweep detection
    tolerance = atr * 0.2
    
    # FVG Detection
    bullish_fvg = low > high.shift(2)
    bearish_fvg = high < low.shift(2)
    
    # FVG size validation
    bullish_fvg_size = bullish_fvg * (low - high.shift(2))
    bearish_fvg_size = bearish_fvg * (low.shift(2) - high)
    
    bullish_fvg_valid = bullish_fvg & (bullish_fvg_size / mintick >= fvg_min_ticks)
    bearish_fvg_valid = bearish_fvg & (bearish_fvg_size / mintick >= fvg_min_ticks)
    
    entries = []
    trade_num = 1
    
    # State variables to track sweeps (simulating var)
    last_high_sweep_bar = None
    last_low_sweep_bar = None
    
    for i in range(len(df)):
        if i < 2:
            continue
        
        if pd.isna(range_high.iloc[i]) or pd.isna(range_low.iloc[i]):
            continue
        
        current_high = high.iloc[i]
        current_low = low.iloc[i]
        current_close = close.iloc[i]
        current_atr = atr.iloc[i]
        current_range_high = range_high.iloc[i]
        current_range_low = range_low.iloc[i]
        
        current_tolerance = current_atr * 0.2
        
        # HIGH SWEEP detection
        if current_high > current_range_high + current_tolerance and current_close < current_range_high:
            last_high_sweep_bar = i
        
        # LOW SWEEP detection
        if current_low < current_range_low - current_tolerance and current_close > current_range_low:
            last_low_sweep_bar = i
        
        # Calculate bars since sweeps
        bars_since_high_sweep = (i - last_high_sweep_bar) if last_high_sweep_bar is not None else 999
        bars_since_low_sweep = (i - last_low_sweep_bar) if last_low_sweep_bar is not None else 999
        
        # Clear old sweeps beyond wait period
        if bars_since_high_sweep > fvg_wait_bars:
            last_high_sweep_bar = None
        if bars_since_low_sweep > fvg_wait_bars:
            last_low_sweep_bar = None
        
        # SHORT Entry: High sweep + Bearish FVG
        short_setup = (bars_since_high_sweep > 0 and bars_since_high_sweep <= fvg_wait_bars)
        if short_setup and bearish_fvg_valid.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(current_close),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(current_close),
                'raw_price_b': float(current_close)
            })
            trade_num += 1
            last_high_sweep_bar = None
        
        # LONG Entry: Low sweep + Bullish FVG
        long_setup = (bars_since_low_sweep > 0 and bars_since_low_sweep <= fvg_wait_bars)
        if long_setup and bullish_fvg_valid.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(current_close),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(current_close),
                'raw_price_b': float(current_close)
            })
            trade_num += 1
            last_low_sweep_bar = None
    
    return entries