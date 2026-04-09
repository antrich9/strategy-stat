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
    if len(df) < 55:
        return []
    
    inp1 = False
    inp2 = False
    inp3 = False
    lookback_bars = 12
    
    ts_col = df['time']
    hour = ((ts_col // 3600000) % 24)
    minute = ((ts_col // 60000) % 60)
    
    in_window = (
        ((hour == 7) & (minute >= 45) & (hour == 7) & (minute < 60)) |
        ((hour >= 8) & (hour < 11)) |
        ((hour == 11) & (minute < 45)) |
        ((hour == 14) & (minute >= 0) & (hour == 14) & (minute < 60)) |
        ((hour >= 15) & (hour < 15)) |
        ((hour == 15) & (minute < 45))
    )
    
    volfilt = df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5
    volfilt = volfilt.fillna(True)
    
    tr = np.maximum(df['high'] - df['low'], 
                    np.maximum(np.abs(df['high'] - df['close'].shift(1)),
                               np.abs(df['low'] - df['close'].shift(1))))
    atr = pd.Series(tr).ewm(alpha=1/20, adjust=False).mean()
    atrfilt = (df['low'] - df['high'].shift(2) > atr / 1.5) | (df['low'].shift(2) - df['high'] > atr / 1.5)
    atrfilt = atrfilt.fillna(True)
    
    loc = df['close'].rolling(54).mean()
    locfiltb = loc > loc.shift(1)
    locfilts = loc <= loc.shift(1)
    locfiltb = locfiltb.fillna(False)
    locfilts = locfilts.fillna(False)
    if not inp3:
        locfiltb = pd.Series(True, index=df.index)
        locfilts = pd.Series(True, index=df.index)
    
    bull_fvg1 = (df['low'] > df['high'].shift(2)) & (df['close'].shift(1) < df['high'].shift(2))
    bear_fvg1 = (df['high'] < df['low'].shift(2)) & (df['close'].shift(1) > df['low'].shift(2))
    
    bull_since = pd.Series(0, index=df.index)
    bear_since = pd.Series(0, index=df.index)
    bull_cond_1 = pd.Series(False, index=df.index)
    bear_cond_1 = pd.Series(False, index=df.index)
    
    for i in range(2, len(df)):
        bull_fvg1_i = bull_fvg1.iloc[i] if not pd.isna(bull_fvg1.iloc[i]) else False
        bear_fvg1_i = bear_fvg1.iloc[i] if not pd.isna(bear_fvg1.iloc[i]) else False
        
        if bull_fvg1_i:
            bull_since.iloc[i] = 0
        elif i > 0:
            bull_since.iloc[i] = bull_since.iloc[i-1] + 1
        
        if bear_fvg1_i:
            bear_since.iloc[i] = 0
        elif i > 0:
            bear_since.iloc[i] = bear_since.iloc[i-1] + 1
        
        bull_cond_1.iloc[i] = bull_fvg1_i and bull_since.iloc[i] <= lookback_bars
        bear_cond_1.iloc[i] = bear_fvg1_i and bear_since.iloc[i] <= lookback_bars
    
    entries = []
    trade_num = 1
    
    for i in range(55, len(df)):
        if pd.isna(loc.iloc[i]):
            continue
        
        in_window_i = in_window.iloc[i] if i < len(in_window) else False
        
        volfilt_i = volfilt.iloc[i] if not pd.isna(volfilt.iloc[i]) else True
        atrfilt_i = atrfilt.iloc[i] if not pd.isna(atrfilt.iloc[i]) else True
        locfiltb_i = locfiltb.iloc[i] if not pd.isna(locfiltb.iloc[i]) else True
        locfilts_i = locfilts.iloc[i] if not pd.isna(locfilts.iloc[i]) else True
        
        bull_cond_1_i = bull_cond_1.iloc[i] if not pd.isna(bull_cond_1.iloc[i]) else False
        bear_cond_1_i = bear_cond_1.iloc[i] if not pd.isna(bear_cond_1.iloc[i]) else False
        
        if in_window_i:
            if bull_cond_1_i and volfilt_i and atrfilt_i and locfiltb_i:
                ts = int(df['time'].iloc[i])
                price = df['close'].iloc[i]
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': price,
                    'raw_price_b': price
                })
                trade_num += 1
            elif bear_cond_1_i and volfilt_i and atrfilt_i and locfilts_i:
                ts = int(df['time'].iloc[i])
                price = df['close'].iloc[i]
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': price,
                    'raw_price_b': price
                })
                trade_num += 1
    
    return entries