import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    lookback = 20
    ret_since = 2
    ret_valid = 2
    bb = lookback
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Calculate Wilder ATR
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # Calculate CMO (Chande Momentum Oscillator)
    momentum = close.diff()
    pos = momentum.where(momentum > 0, 0.0)
    neg = momentum.where(momentum < 0, 0.0).abs()
    pos_sum = pos.ewm(alpha=1/14, adjust=False).mean()
    neg_sum = neg.ewm(alpha=1/14, adjust=False).mean()
    cmo = ((pos_sum - neg_sum) / (pos_sum + neg_sum)) * 100
    
    # Pivot calculations
    pl = low.rolling(window=2*bb+1, center=True).min()
    ph = high.rolling(window=2*bb+1, center=True).max()
    pl = pl.shift(bb)
    ph = ph.shift(bb)
    
    # Box heights
    s_yLoc = pd.Series(np.where(low.shift(bb + 1) > low.shift(bb - 1), low.shift(bb - 1), low.shift(bb + 1)), index=df.index)
    r_yLoc = pd.Series(np.where(high.shift(bb + 1) > high.shift(bb - 1), high.shift(bb + 1), high.shift(bb - 1)), index=df.index)
    
    sBot = pl
    rTop = ph
    
    sBreak = False
    rBreak = False
    sRetOccurred = False
    rRetOccurred = False
    
    entries = []
    trade_num = 1
    tradeDirection = "Both"
    
    for i in range(len(df)):
        if pd.notna(pl.iloc[i]) and pd.notna(pl.iloc[i-1]) and pl.iloc[i] != pl.iloc[i-1]:
            if sBreak:
                sBreak = False
        if pd.notna(ph.iloc[i]) and pd.notna(ph.iloc[i-1]) and ph.iloc[i] != ph.iloc[i-1]:
            if rBreak:
                rBreak = False
        
        sTop_val = pl.iloc[i] if pd.notna(pl.iloc[i]) else None
        sBot_val = pl.iloc[i] if pd.notna(pl.iloc[i]) else None
        rTop_val = ph.iloc[i] if pd.notna(ph.iloc[i]) else None
        rBot_val = ph.iloc[i] if pd.notna(ph.iloc[i]) else None
        
        if sTop_val is not None:
            cu = close.iloc[i] < sBot_val and close.iloc[i-1] >= sBot_val if i > 0 else False
            if cu and not sBreak:
                sBreak = True
        if rTop_val is not None:
            co = close.iloc[i] > rTop_val and close.iloc[i-1] <= rTop_val if i > 0 else False
            if co and not rBreak:
                rBreak = True
        
        bars_since_sBreak = 0
        bars_since_rBreak = 0
        if sBreak:
            for j in range(i-1, -1, -1):
                if not sBreak:
                    break
                bars_since_sBreak += 1
        if rBreak:
            for j in range(i-1, -1, -1):
                if not rBreak:
                    break
                bars_since_rBreak += 1
        
        sRetEvent = sBreak and bars_since_sBreak > ret_since
        rRetEvent = rBreak and bars_since_rBreak > ret_since
        
        sRetValid = False
        rRetValid = False
        
        if sRetEvent:
            s1 = high.iloc[i] >= sTop_val and close.iloc[i] <= sBot_val
            s2 = high.iloc[i] >= sTop_val and close.iloc[i] >= sBot_val and close.iloc[i] <= sTop_val
            s3 = high.iloc[i] >= sBot_val and high.iloc[i] <= sTop_val
            s4 = high.iloc[i] >= sBot_val and high.iloc[i] <= sTop_val and close.iloc[i] < sBot_val
            if s1 or s2 or s3 or s4:
                ret_valid_count = 0
                for j in range(i-1, -1, -1):
                    s1j = high.iloc[j] >= sTop_val and close.iloc[j] <= sBot_val
                    s2j = high.iloc[j] >= sTop_val and close.iloc[j] >= sBot_val and close.iloc[j] <= sTop_val
                    s3j = high.iloc[j] >= sBot_val and high.iloc[j] <= sTop_val
                    s4j = high.iloc[j] >= sBot_val and high.iloc[j] <= sTop_val and close.iloc[j] < sBot_val
                    if s1j or s2j or s3j or s4j:
                        ret_valid_count += 1
                        if ret_valid_count >= ret_valid:
                            break
                if ret_valid_count > 0 and ret_valid_count <= ret_valid:
                    sRetValid = True
        
        if rRetEvent:
            r1 = low.iloc[i] <= rBot_val and close.iloc[i] >= rTop_val
            r2 = low.iloc[i] <= rBot_val and close.iloc[i] <= rTop_val and close.iloc[i] >= rBot_val
            r3 = low.iloc[i] <= rTop_val and low.iloc[i] >= rBot_val
            r4 = low.iloc[i] <= rTop_val and low.iloc[i] >= rBot_val and close.iloc[i] > rTop_val
            if r1 or r2 or r3 or r4:
                ret_valid_count = 0
                for j in range(i-1, -1, -1):
                    r1j = low.iloc[j] <= rBot_val and close.iloc[j] >= rTop_val
                    r2j = low.iloc[j] <= rBot_val and close.iloc[j] <= rTop_val and close.iloc[j] >= rBot_val
                    r3j = low.iloc[j] <= rTop_val and low.iloc[j] >= rBot_val
                    r4j = low.iloc[j] <= rTop_val and low.iloc[j] >= rBot_val and close.iloc[j] > rTop_val
                    if r1j or r2j or r3j or r4j:
                        ret_valid_count += 1
                        if ret_valid_count >= ret_valid:
                            break
                if ret_valid_count > 0 and ret_valid_count <= ret_valid:
                    rRetValid = True
        
        if sRetValid and not sRetOccurred and tradeDirection in ["Long", "Both"]:
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
            sRetOccurred = True
            sBreak = False
        
        if rRetValid and not rRetOccurred and tradeDirection in ["Short", "Both"]:
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
            rRetOccurred = True
            rBreak = False
    
    return entries