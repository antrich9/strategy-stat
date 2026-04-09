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
    trade_num = 1
    
    # Parameters from Pine Script
    Swing_Length = 7
    TF_Multip = 2
    TF_Period = "Hour"
    Use_High_Low = False
    Chart_Feature = False
    Use_Bos_Plot = True
    Use_Mss_Plot = True
    
    # Time window settings (London session)
    def is_within_london_window(timestamp_ms):
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        total_minutes = hour * 60 + minute
        morning_start = 8 * 60
        morning_end = 9 * 60 + 55
        afternoon_start = 14 * 60
        afternoon_end = 16 * 60 + 55
        return (morning_start <= total_minutes < morning_end) or (afternoon_start <= total_minutes < afternoon_end)
    
    n = len(df)
    if n < Swing_Length * 2 + 1:
        return entries
    
    # Calculate pivothigh and pivotlow
    pivothigh = pd.Series(index=df.index, dtype=float)
    pivotlow = pd.Series(index=df.index, dtype=float)
    
    for i in range(Swing_Length, n - Swing_Length):
        is_high = True
        is_low = True
        for j in range(1, Swing_Length + 1):
            if df['high'].iloc[i] <= df['high'].iloc[i + j]:
                is_high = False
            if df['low'].iloc[i] >= df['low'].iloc[i - j]:
                is_low = False
        if is_high:
            pivothigh.iloc[i] = df['high'].iloc[i]
        if is_low:
            pivotlow.iloc[i] = df['low'].iloc[i]
    
    # Initialize market structure variables
    prev_high = np.nan
    prev_low = np.nan
    prev_high_time = 0
    prev_low_time = 0
    high_present = False
    low_present = False
    prev_breakout_type = 0
    
    hh = False
    lh = False
    hl = False
    ll = False
    high_broken = False
    low_broken = False
    end_high_time = 0
    end_low_time = 0
    
    high_count = 0
    low_count = 0
    
    bullish_mss_signal = False
    bearish_mss_signal = False
    last_mss_type = None
    
    # Iterate through bars
    for i in range(n):
        ts = df['time'].iloc[i]
        in_trading_window = is_within_london_window(ts)
        
        if not in_trading_window:
            continue
        
        sh = pivothigh.iloc[i] if not pd.isna(pivothigh.iloc[i]) else np.nan
        sl = pivotlow.iloc[i] if not pd.isna(pivotlow.iloc[i]) else np.nan
        
        # Market structure logic
        if not pd.isna(sh):
            if sh >= prev_high:
                hh = True
                lh = False
            else:
                lh = True
                hh = False
            prev_high = sh
            prev_high_time = ts
            high_present = True

        if not pd.isna(sl):
            if sl >= prev_low:
                hl = True
                ll = False
            else:
                ll = True
                hl = False
            prev_low = sl
            prev_low_time = ts
            low_present = True
        
        # High/Low close price for break detection
        high_close_price = df['high'].iloc[i] if Use_High_Low else df['close'].iloc[i]
        low_close_price = df['low'].iloc[i] if Use_High_Low else df['close'].iloc[i]
        
        if high_close_price > prev_high and high_present:
            high_broken = True
            high_present = False
            end_high_time = ts
            
        if low_close_price < prev_low and low_present:
            low_broken = True
            low_present = False
            end_low_time = ts
        
        # Calculate bars since
        if hh or lh:
            high_count = 0
        else:
            high_count += 1
            
        if hl or ll:
            low_count = 0
        else:
            low_count += 1
        
        # Generate signals based on market structure breaks
        if high_broken and not high_present:
            if prev_breakout_type == 1 and Use_Bos_Plot:
                bullish_mss_signal = False
                bearish_mss_signal = True
            elif prev_breakout_type == -1 and Use_Mss_Plot:
                bullish_mss_signal = False
                bearish_mss_signal = False
                last_mss_type = "bullish"
            
            if prev_breakout_type == -1 and Use_Mss_Plot:
                # Bullish MSS entry
                entry_price = df['close'].iloc[i]
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(ts),
                    'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
            
            prev_breakout_type = 1
            high_broken = False
        
        if low_broken and not low_present:
            if prev_breakout_type == -1 and Use_Bos_Plot:
                bearish_mss_signal = True
                bullish_mss_signal = False
            elif prev_breakout_type == 1 and Use_Mss_Plot:
                bearish_mss_signal = False
                bullish_mss_signal = False
                last_mss_type = "bearish"
            
            if prev_breakout_type == 1 and Use_Mss_Plot:
                # Bearish MSS entry
                entry_price = df['close'].iloc[i]
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(ts),
                    'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
            
            prev_breakout_type = -1
            low_broken = False
    
    return entries