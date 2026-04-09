import pandas as pd
import numpy as np
from datetime import datetime, timezone

def calculate_wilder_atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

def calculate_mcginley_dynamic(close: pd.Series, length: int) -> pd.Series:
    mgd = pd.Series(index=close.index, dtype=float)
    mgd.iloc[0] = close.iloc[0]
    
    for i in range(1, len(close)):
        prev_mgd = mgd.iloc[i-1]
        curr_close = close.iloc[i]
        if prev_mgd != 0:
            mgd.iloc[i] = prev_mgd + (curr_close - prev_mgd) / (length * ((curr_close / prev_mgd) ** 4))
        else:
            mgd.iloc[i] = curr_close
    
    return mgd

def calculate_doda_stochastic(df: pd.DataFrame, length: int, smooth_k: int, smooth_d: int):
    high = df['high']
    low = df['low']
    close = df['close']
    
    lowest_low = low.rolling(window=length).min()
    highest_high = high.rolling(window=length).max()
    
    stoch = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = stoch.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    
    return k, d

def calculate_pivot_low(low: pd.Series, bb: int):
    pivot_low = low.rolling(window=bb*2+1).min()
    pivot_low = pivot_low.where(pivot_low == low.shift(bb), np.nan)
    pivot_low = pivot_low.fillna(method='ffill')
    return pivot_low

def calculate_pivot_high(high: pd.Series, bb: int):
    pivot_high = high.rolling(window=bb*2+1).max()
    pivot_high = pivot_high.where(pivot_high == high.shift(bb), np.nan)
    pivot_high = pivot_high.fillna(method='ffill')
    return pivot_high

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
    atrLength = 14
    atrMultiplier = 2
    lengthMD = 10
    dodaLength = 14
    dodaSmoothK = 3
    dodaSmoothD = 3
    overboughtLevel = 80
    oversoldLevel = 20
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    atr = calculate_wilder_atr(df, atrLength)
    mgd = calculate_mcginley_dynamic(close, lengthMD)
    k, d = calculate_doda_stochastic(df, dodaLength, dodaSmoothK, dodaSmoothD)
    
    bb = input_lookback
    
    pl = calculate_pivot_low(low, bb)
    ph = calculate_pivot_high(high, bb)
    
    pl_change = pl.diff() != 0
    ph_change = ph.diff() != 0
    
    s_yLoc = np.where(low.shift(bb + 1) > low.shift(bb - 1), low.shift(bb - 1), low.shift(bb + 1))
    r_yLoc = np.where(high.shift(bb + 1) > high.shift(bb - 1), high.shift(bb + 1), high.shift(bb - 1))
    
    sBot = pd.Series(np.nan, index=df.index)
    sTop = pd.Series(np.nan, index=df.index)
    rBot = pd.Series(np.nan, index=df.index)
    rTop = pd.Series(np.nan, index=df.index)
    
    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)
    
    sRetValid = pd.Series(False, index=df.index)
    rRetValid = pd.Series(False, index=df.index)
    
    for i in range(bb + 2, len(df)):
        if i >= 1 and pl_change.iloc[i]:
            sBot_i = s_yLoc[i - bb - 1] if not np.isnan(s_yLoc[i - bb - 1]) else low.iloc[i - bb - 1]
            sTop_i = pl.iloc[i - bb - 1] if not np.isnan(pl.iloc[i - bb - 1]) else low.iloc[i - bb - 1]
            sBot.iloc[i] = sBot_i
            sTop.iloc[i] = sTop_i
            if pd.notna(sBreak.iloc[i]) and sBreak.iloc[i]:
                sBreak.iloc[i] = False
            else:
                sBreak.iloc[i] = False
        elif i >= 1:
            sBot.iloc[i] = sBot.iloc[i-1]
            sTop.iloc[i] = sTop.iloc[i-1]
            sBreak.iloc[i] = sBreak.iloc[i-1]
        
        if i >= 1 and ph_change.iloc[i]:
            rBot_i = r_yLoc[i - bb - 1] if not np.isnan(r_yLoc[i - bb - 1]) else high.iloc[i - bb - 1]
            rTop_i = ph.iloc[i - bb - 1] if not np.isnan(ph.iloc[i - bb - 1]) else high.iloc[i - bb - 1]
            rBot.iloc[i] = rBot_i
            rTop.iloc[i] = rTop_i
            if pd.notna(rBreak.iloc[i]) and rBreak.iloc[i]:
                rBreak.iloc[i] = False
            else:
                rBreak.iloc[i] = False
        elif i >= 1:
            rBot.iloc[i] = rBot.iloc[i-1]
            rTop.iloc[i] = rTop.iloc[i-1]
            rBreak.iloc[i] = rBreak.iloc[i-1]
    
    cu = (close < sBot) & (close.shift(1) >= sBot) & (low < sBot) & (low.shift(1) >= sBot)
    co = (close > rTop) & (close.shift(1) <= rTop) & (high > rTop) & (high.shift(1) <= rTop)
    
    for i in range(bb + 2, len(df)):
        if cu.iloc[i] and not sBreak.iloc[i]:
            sBreak.iloc[i] = True
        
        if co.iloc[i] and not rBreak.iloc[i]:
            rBreak.iloc[i] = True
        
        if pl_change.iloc[i]:
            sBreak.iloc[i] = False
        
        if ph_change.iloc[i]:
            rBreak.iloc[i] = False
    
    sBreak = sBreak.fillna(False)
    rBreak = rBreak.fillna(False)
    
    s1 = (close.shift(input_retSince) > 0) & (high >= sTop) & (close <= sBot)
    s2 = (close.shift(input_retSince) > 0) & (high >= sTop) & (close >= sBot) & (close <= sTop)
    s3 = (close.shift(input_retSince) > 0) & (high >= sBot) & (high <= sTop)
    s4 = (close.shift(input_retSince) > 0) & (high >= sBot) & (high <= sTop) & (close < sBot)
    
    r1 = (close.shift(input_retSince) > 0) & (low <= rBot) & (close >= rTop)
    r2 = (close.shift(input_retSince) > 0) & (low <= rBot) & (close <= rTop) & (close >= rBot)
    r3 = (close.shift(input_retSince) > 0) & (low <= rTop) & (low >= rBot)
    r4 = (close.shift(input_retSince) > 0) & (low <= rTop) & (low >= rBot) & (close > rTop)
    
    sRetEvent = s1 | s2 | s3 | s4
    rRetEvent = r1 | r2 | r3 | r4
    
    sRetValid = pd.Series(False, index=df.index)
    rRetValid = pd.Series(False, index=df.index)
    
    retOccurred_s = False
    retOccurred_r = False
    
    for i in range(bb + 2, len(df)):
        if sRetEvent.iloc[i]:
            retOccurred_s = False
        
        if rRetEvent.iloc[i]:
            retOccurred_r = False
        
        bars_since_s = 0
        bars_since_r = 0
        
        if sRetEvent.iloc[i]:
            bars_since_s = 0
        else:
            for j in range(1, input_retValid + 2):
                if i - j >= 0 and sRetEvent.iloc[i - j]:
                    bars_since_s = j
                    break
        
        if rRetEvent.iloc[i]:
            bars_since_r = 0
        else:
            for j in range(1, input_retValid + 2):
                if i - j >= 0 and rRetEvent.iloc[i - j]:
                    bars_since_r = j
                    break
        
        if bars_since_s > 0 and bars_since_s <= input_retValid:
            retValid_s = (close.iloc[i] <= sBot.iloc[i] if i - bars_since_s >= 0 else False) and not retOccurred_s
            if retValid_s:
                sRetValid.iloc[i] = True
                retOccurred_s = True
        
        if bars_since_r > 0 and bars_since_r <= input_retValid:
            retValid_r = (close.iloc[i] >= rTop.iloc[i] if i - bars_since_r >= 0 else False) and not retOccurred_r
            if retValid_r:
                rRetValid.iloc[i] = True
                retOccurred_r = True
    
    trade_num = 1
    entries = []
    
    for i in range(len(df)):
        if i < bb + 2:
            continue
        
        if pd.isna(k.iloc[i]) or pd.isna(d.iloc[i]) or pd.isna(atr.iloc[i]):
            continue
        
        if sRetValid.iloc[i] and k.iloc[i] > overboughtLevel:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        
        if rRetValid.iloc[i] and k.iloc[i] < oversoldLevel:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
    
    return entries