import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    multiplier = 3.0
    atrLength = 10
    
    # True Range
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - prev_close)
    tr3 = np.abs(df['low'] - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Initialize ATR with Wilder's method
    atr = pd.Series(np.nan, index=df.index)
    first_valid_idx = atrLength - 1
    
    # Simple average for first ATR
    atr.iloc[first_valid_idx] = tr.iloc[:atrLength].mean()
    
    # Wilder smoothing for subsequent ATR values
    for i in range(atrLength, len(df)):
        atr.iloc[i] = (atr.iloc[i-1] * (atrLength - 1) + tr.iloc[i]) / atrLength
    
    # Supertrend calculation
    hl2 = (df['high'] + df['low']) / 2
    trends = pd.Series(np.nan, index=df.index)
    upper_bands = pd.Series(np.nan, index=df.index)
    lower_bands = pd.Series(np.nan, index=df.index)
    
    # Initialize first valid values
    first_valid = first_valid_idx
    atr_val = atr.iloc[first_valid]
    hl2_val = hl2.iloc[first_valid]
    
    lower_band = hl2_val - multiplier * atr_val
    upper_band = hl2_val + multiplier * atr_val
    
    # Check if first close is above or below initial bands
    first_close = df['close'].iloc[first_valid]
    if first_close >= lower_band:
        trend_value = 1
    else:
        trend_value = -1
    
    trends.iloc[first_valid] = trend_value
    upper_bands.iloc[first_valid] = upper_band
    lower_bands.iloc[first_valid] = lower_band
    
    # Calculate Supertrend for remaining bars
    for i in range(first_valid + 1, len(df)):
        atr_val = atr.iloc[i]
        hl2_val = hl2.iloc[i]
        prev_trend = trends.iloc[i-1]
        prev_upper = upper_bands.iloc[i-1]
        prev_lower = lower_bands.iloc[i-1]
        prev_close = df['close'].iloc[i-1]
        curr_close = df['close'].iloc[i]
        
        if prev_trend == 1:
            lower_band = hl2_val - multiplier * atr_val
            if curr_close < prev_upper:
                trend_value = -1
                upper_band = hl2_val + multiplier * atr_val
            else:
                trend_value = 1
                upper_band = prev_upper
        else:
            upper_band = hl2_val + multiplier * atr_val
            if curr_close > prev_lower:
                trend_value = 1
                lower_band = hl2_val - multiplier * atr_val
            else:
                trend_value = -1
                lower_band = prev_lower
        
        trends.iloc[i] = trend_value
        upper_bands.iloc[i] = upper_band
        lower_bands.iloc[i] = lower_band
    
    # Entry conditions
    longCondition = (trends == 1) & (trends.shift(1).fillna(0) <= 0)
    shortCondition = (trends == -1) & (trends.shift(1).fillna(0) >= 0)
    
    # Build entry list
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if pd.isna(trends.iloc[i]) or pd.isna(trends.iloc[i-1]):
            continue
            
        if longCondition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_price_guess = float(df['close'].iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1
        elif shortCondition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_price_guess = float(df['close'].iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1
    
    return entries