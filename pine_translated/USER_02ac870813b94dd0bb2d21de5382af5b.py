def generate_entries(df: pd.DataFrame) -> list:
    import pandas as pd
    import numpy as np
    from datetime import datetime, timezone
    
    # Initialize state variables
    bull_fvg_upper = pd.Series(np.nan, index=df.index)
    bull_fvg_lower = pd.Series(np.nan, index=df.index)
    bear_fvg_upper = pd.Series(np.nan, index=df.index)
    bear_fvg_lower = pd.Series(np.nan, index=df.index)
    last = pd.Series(False, index=df.index)
    fvg_size = pd.Series(0, index=df.index)
    fvg_last_pct = pd.Series(0.0, index=df.index)
    
    prev_day_high = pd.Series(np.nan, index=df.index)
    prev_day_low = pd.Series(np.nan, index=df.index)
    flagpdl = pd.Series(False, index=df.index)
    flagpdh = pd.Series(False, index=df.index)
    
    bull_midpoint = pd.Series(np.nan, index=df.index)
    bear_midpoint = pd.Series(np.nan, index=df.index)
    
    bull_pre_entry = pd.Series(False, index=df.index)
    bear_pre_entry = pd.Series(False, index=df.index)
    
    atr_144 = calculate_atr(df, 144)
    bull_gap = df['low'] > df['high'].shift(1)
    bear_gap = df['high'] < df['low'].shift(1)