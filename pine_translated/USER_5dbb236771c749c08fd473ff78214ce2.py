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
    lookback_bars = 12
    threshold = 0.0
    
    entries = []
    trade_num = 1
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Detect FVG (Fair Value Gap) patterns
    bull_fvg = (low > high.shift(2)) & (close.shift(1) > high.shift(2))
    bear_fvg = (high < low.shift(2)) & (close.shift(1) < low.shift(2))
    
    bull_since = pd.Series(index=df.index, dtype=float)
    bear_since = pd.Series(index=df.index, dtype=float)
    
    bull_result = pd.Series(False, index=df.index)
    bear_result = pd.Series(False, index=df.index)
    
    bull_fvg_top = pd.Series(np.nan, index=df.index)
    bull_fvg_bottom = pd.Series(np.nan, index=df.index)
    bear_fvg_top = pd.Series(np.nan, index=df.index)
    bear_fvg_bottom = pd.Series(np.nan, index=df.index)
    
    active_bull_fvgs = {}  # timestamp -> {'top': float, 'bottom': float, 'bar_idx': int, 'detected_bar': int}
    active_bear_fvgs = {}
    detected_bull_fvgs = {}  # bar_idx -> {'top': float, 'bottom': float}
    detected_bear_fvgs = {}
    
    for i in range(2, len(df)):
        # Calculate bars since for FVG detection
        if i > 0:
            if bull_fvg.iloc[i]:
                bull_since.iloc[i] = 0
            elif i > 0 and not pd.isna(bull_since.iloc[i-1]):
                bull_since.iloc[i] = bull_since.iloc[i-1] + 1
            else:
                bull_since.iloc[i] = np.nan
                
            if bear_fvg.iloc[i]:
                bear_since.iloc[i] = 0
            elif i > 0 and not pd.isna(bear_since.iloc[i-1]):
                bear_since.iloc[i] = bear_since.iloc[i-1] + 1
            else:
                bear_since.iloc[i] = np.nan
        
        # Bull FVG result calculation
        if bull_fvg.iloc[i] and not pd.isna(bull_since.iloc[i]) and bull_since.iloc[i] <= lookback_bars:
            bar_back = int(bull_since.iloc[i])
            combined_low_val = max(high.iloc[i - bar_back], high.iloc[i - 2]) if i - bar_back >= 0 and i - 2 >= 0 else np.nan
            combined_high_val = min(low.iloc[i - bar_back + 2], low.iloc[i]) if i - bar_back + 2 < len(df) else np.nan
            
            if not pd.isna(combined_low_val) and not pd.isna(combined_high_val) and (combined_high_val - combined_low_val >= threshold):
                bull_result.iloc[i] = True
                bull_fvg_top.iloc[i] = combined_high_val
                bull_fvg_bottom.iloc[i] = combined_low_val
                detected_bull_fvgs[i] = {'top': combined_high_val, 'bottom': combined_low_val, 'detected_bar': i}
        
        # Bear FVG result calculation
        if bear_fvg.iloc[i] and not pd.isna(bear_since.iloc[i]) and bear_since.iloc[i] <= lookback_bars:
            bar_back = int(bear_since.iloc[i])
            combined_low_val = max(high.iloc[i - bar_back + 2], high.iloc[i]) if i - bar_back + 2 < len(df) else np.nan
            combined_high_val = min(low.iloc[i - bar_back], low.iloc[i - 2]) if i - bar_back >= 0 and i - 2 >= 0 else np.nan
            
            if not pd.isna(combined_low_val) and not pd.isna(combined_high_val) and (combined_high_val - combined_low_val >= threshold):
                bear_result.iloc[i] = True
                bear_fvg_top.iloc[i] = combined_high_val
                bear_fvg_bottom.iloc[i] = combined_low_val
                detected_bear_fvgs[i] = {'top': combined_high_val, 'bottom': combined_low_val, 'detected_bar': i}
        
        # Update active FVG tracking for entry detection
        for bar_idx, fvg_data in list(detected_bull_fvgs.items()):
            if bar_idx <= i and i - bar_idx <= lookback_bars:
                if bar_idx not in active_bull_fvgs:
                    active_bull_fvgs[bar_idx] = {'top': fvg_data['top'], 'bottom': fvg_data['bottom'], 'bar_idx': bar_idx, 'detected_bar': i}
        
        for bar_idx, fvg_data in list(detected_bear_fvgs.items()):
            if bar_idx <= i and i - bar_idx <= lookback_bars:
                if bar_idx not in active_bear_fvgs:
                    active_bear_fvgs[bar_idx] = {'top': fvg_data['top'], 'bottom': fvg_data['bottom'], 'bar_idx': bar_idx, 'detected_bar': i}
        
        # Check for bullish entry (price enters bull FVG from above - low crosses below FVG top)
        if bull_result.iloc[i]:
            fvg_top = bull_fvg_top.iloc[i]
            if not pd.isna(fvg_top) and low.iloc[i] < fvg_top:
                entry_ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(fvg_top),
                    'raw_price_b': float(fvg_top)
                })
                trade_num += 1
        
        # Check for bearish entry (price enters bear FVG from below - high crosses above FVG bottom)
        if bear_result.iloc[i]:
            fvg_bottom = bear_fvg_bottom.iloc[i]
            if not pd.isna(fvg_bottom) and high.iloc[i] > fvg_bottom:
                entry_ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(fvg_bottom),
                    'raw_price_b': float(fvg_bottom)
                })
                trade_num += 1
        
        # Clean up old active FVGs beyond lookback
        for bar_idx in list(active_bull_fvgs.keys()):
            if i - bar_idx > lookback_bars:
                del active_bull_fvgs[bar_idx]
        
        for bar_idx in list(active_bear_fvgs.keys()):
            if i - bar_idx > lookback_bars:
                del active_bear_fvgs[bar_idx]
    
    return entries