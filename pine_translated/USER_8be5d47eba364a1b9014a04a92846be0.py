import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # SMA for local filter (loc = ta.sma(close, 54))
    loc = close.rolling(54).mean()
    
    # Volume filter - ta.sma(volume, 9)
    vol_sma = volume.rolling(9).mean()
    volfilt = volume.shift(1) > vol_sma * 1.5
    
    # ATR - ta.atr(20) / 1.5
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_raw = tr.ewm(alpha=1/20, adjust=False).mean()
    atr = atr_raw / 1.5
    
    # ATR filter
    atrfilt = (low - high.shift(2) > atr) | (low.shift(2) - high > atr)
    
    # Local filter conditions
    locfiltb = loc > loc.shift(1)
    locfilts = loc <= loc.shift(1)
    
    # FVG conditions
    bfvg = (low > high.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (high < low.shift(2)) & volfilt & atrfilt & locfilts
    
    # Pivot high and low
    pivot_high = pd.Series(np.nan, index=df.index)
    pivot_low = pd.Series(np.nan, index=df.index)
    
    for i in range(5, len(df)):
        left_high = high.iloc[i-5:i+1].max()
        if high.iloc[i-5] == left_high:
            right_max = high.iloc[i-4:i+1].max() if i >= 4 else high.iloc[max(0, i-4):i+1].max()
            if right_max < left_high:
                pivot_high.iloc[i] = high.iloc[i-5]
        
        left_low = low.iloc[i-5:i+1].min()
        if low.iloc[i-5] == left_low:
            right_min = low.iloc[i-4:i+1].min() if i >= 4 else low.iloc[max(0, i-4):i+1].min()
            if right_min > left_low:
                pivot_low.iloc[i] = low.iloc[i-5]
    
    # Entry signals
    long_cond = bfvg & pivot_high.notna()
    short_cond = sfvg & pivot_low.notna()
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(loc.iloc[i]) or pd.isna(atr.iloc[i]):
            continue
        
        if long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        
        if short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
    
    return entries