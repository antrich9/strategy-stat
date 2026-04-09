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
    # Parameters from Pine Script
    PP = 6  # Pivot Period of Order Blocks Detector
    ATR_LEN = 55  # ATR length for filters
    
    # VP Trap Filter parameters
    VP_BINS = 40
    VP_VA_PCT = 70
    VP_ATR_MULT = 0.3
    
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # Calculate Wilder ATR(55)
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = true_range.ewm(alpha=1.0/ATR_LEN, adjust=False).mean()
    
    # Calculate Pivot Highs and Lows (ta.pivothigh/ta.pivotlow)
    pivot_high = pd.Series(np.nan, index=high.index)
    pivot_low = pd.Series(np.nan, index=low.index)
    
    for i in range(PP, len(high) - PP):
        # Pivot high: highest high in left PP bars and right PP bars
        left_window = high.iloc[i-PP:i+1]
        right_window = high.iloc[i:i+PP+1]
        if high.iloc[i] == left_window.max() and high.iloc[i] == right_window.max():
            pivot_high.iloc[i] = high.iloc[i]
        
        # Pivot low: lowest low in left PP bars and right PP bars
        left_window_l = low.iloc[i-PP:i+1]
        right_window_l = low.iloc[i:i+PP+1]
        if low.iloc[i] == left_window_l.min() and low.iloc[i] == right_window_l.min():
            pivot_low.iloc[i] = low.iloc[i]
    
    # Track Major/Minor levels for market structure
    major_high = pd.Series(np.nan, index=high.index)
    major_low = pd.Series(np.nan, index=low.index)
    minor_high = pd.Series(np.nan, index=high.index)
    minor_low = pd.Series(np.nan, index=low.index)
    
    # State variables for structure tracking
    last_major_high_idx = 0
    last_major_low_idx = 0
    last_minor_high_idx = 0
    last_minor_low_idx = 0
    
    # Volume Profile calculation for VP Trap Filter
    def calc_vp_filter(i, start_idx, end_idx):
        if end_idx <= start_idx or start_idx < 0:
            return False
        end_idx = min(end_idx, i)
        if end_idx <= start_idx:
            return False
        
        lo = low.iloc[start_idx:end_idx+1].min()
        hi = high.iloc[start_idx:end_idx+1].max()
        
        if hi <= lo:
            return False
        
        bin_size = (hi - lo) / VP_BINS
        vol_bins = np.zeros(VP_BINS)
        
        for j in range(start_idx, end_idx+1):
            bar_vol = volume.iloc[j]
            if bar_vol > 0:
                bar_hi = high.iloc[j]
                bar_lo = low.iloc[j]
                top_bin = min(VP_BINS - 1, int((bar_hi - lo) / bin_size))
                bot_bin = max(0, int((bar_lo - lo) / bin_size))
                for b in range(bot_bin, top_bin + 1):
                    bin_lo = lo + b * bin_size