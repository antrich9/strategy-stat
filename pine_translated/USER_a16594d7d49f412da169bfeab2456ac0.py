import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Constants from the strategy
    htf_period = 60  # minutes
    fvg_tf1_period = 5  # minutes
    fvg_tf2_period = 15  # minutes
    lookback_bars = 500
    max_ifvg_candles = 6
    min_ifvg_atr_mult = 0.1
    
    # HTF bar size in current TF bars
    htf_bar_size = htf_period // fvg_tf1_period  # 60/5 = 12
    fvg_tf2_bar_size = fvg_tf2_period // fvg_tf1_period  # 15/5 = 3
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_arr = df['open'].values
    n = len(df)
    
    # 1. Calculate ATR (Wilder)
    atr = calculate_wilder_atr(df, 14)
    
    # 2. Build HTF bars
    htf_bar_index = np.arange(n) // htf_bar_size
    
    # For each HTF bar, get the range of 5m indices
    htf_groups = pd.core.groupby.ops.GroupBy.GroupBy(pd.Series(np.arange(n)), htf_bar_index)