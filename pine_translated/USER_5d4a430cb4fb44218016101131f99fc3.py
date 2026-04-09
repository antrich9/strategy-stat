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
    emaF = 8
    emaS = 21
    voSLen = 5
    voLLen = 10
    zlF = 12
    zlS = 26
    zlSig = 9
    ttmsLen = 20
    
    # Calculate indicators
    
    # EMA Cloud Baseline
    ema_fast = df['close'].ewm(span=emaF, adjust=False).mean()
    ema_slow = df['close'].ewm(span=emaS, adjust=False).mean()
    
    # Baseline conditions
    baseline_long = ema_fast > ema_slow
    baseline_short = ema_fast < ema_slow
    
    # Volume Oscillator
    vo_short = df['volume'].ewm(span=voSLen, adjust=False).mean()
    vo_long = df['volume'].ewm(span=voLLen, adjust=False).mean()
    volume_osc = ((vo_short - vo_long) / vo_long) * 100
    
    volume_long = volume_osc > 0
    volume_short = volume_osc < 0
    
    # Zero Lag MACD (simplified)
    ema_zl_fast = df['close'].ewm(span=zlF, adjust=False).mean()
    ema_zl_sift = df['close'].ewm(span=zlS, adjust=False).mean()
    zl_macd_line = ema_zl_fast - ema_zl_sift
    zl_signal = zl_macd_line.ewm(span=zlSig, adjust=False).mean()
    
    confirm1_long = zl_macd_line > zl_signal
    confirm1_short = zl_macd_line < zl_signal
    
    # TTMS (Triple Triple Moving System)
    ttms_ema1 = df['close'].ewm(span=ttmsLen, adjust=False).mean()
    ttms_ema2 = ttms_ema1.ewm(span=ttmsLen, adjust=False).mean()
    ttms_ema3 = ttms_ema2.ewm(span=ttmsLen, adjust=False).mean()
    ttms_value = 3 * ttms_ema1 - 3 * ttms_ema2 + ttms_ema3
    
    confirm2_long = df['close'] > ttms_value
    confirm2_short = df['close'] < ttms_value
    
    # Entry signals (All Must Agree mode)
    entries_long = baseline_long & volume_long & confirm1_long & confirm2_long
    entries_short = baseline_short & volume_short & confirm1_short & confirm2_short
    
    # Generate trade list
    trades = []
    trade_num = 1
    
    n = len(df)
    for i in range(1, n):
        if pd.isna(ema_fast.iloc[i]) or pd.isna(volume_osc.iloc[i]) or pd.isna(ttms_value.iloc[i]):
            continue
            
        direction = None
        
        if entries_long.iloc[i] and not entries_long.iloc[i-1]:
            direction = 'long'
        elif entries_short.iloc[i] and not entries_short.iloc[i-1]:
            direction = 'short'
        
        if direction:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
            trades.append({
                'trade_num': trade_num,
                'direction': direction,
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
    
    return trades