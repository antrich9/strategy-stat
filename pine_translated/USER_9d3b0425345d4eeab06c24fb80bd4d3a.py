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
    pp = 5
    atr_len = 55
    
    high = df['high']
    low = df['low']
    close = df['close']
    n = len(df)
    
    # Wilder ATR
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/atr_len, adjust=False).mean()
    
    # Pivot detection
    pivothigh = pd.Series(False, index=df.index)
    pivotlow = pd.Series(False, index=df.index)
    ph_value = pd.Series(np.nan, index=df.index)
    pl_value = pd.Series(np.nan, index=df.index)
    
    for i in range(pp, n - pp):
        is_h = True
        for j in range(1, pp + 1):
            if high.iloc[i] <= high.iloc[i - j] or high.iloc[i] <= high.iloc[i + j]:
                is_h = False
                break
        if is_h:
            pivothigh.iloc[i] = True
            ph_value.iloc[i] = high.iloc[i]
        
        is_l = True
        for j in range(1, pp + 1):
            if low.iloc[i] >= low.iloc[i - j] or low.iloc[i] >= low.iloc[i + j]:
                is_l = False
                break
        if is_l:
            pivotlow.iloc[i] = True
            pl_value.iloc[i] = low.iloc[i]
    
    # Market structure tracking
    major_high = pd.Series(np.nan, index=df.index)
    major_low = pd.Series(np.nan, index=df.index)
    major_h_idx = pd.Series(np.nan, index=df.index)
    major_l_idx = pd.Series(np.nan, index=df.index)
    bull_bos = pd.Series(False, index=df.index)
    bear_bos = pd.Series(False, index=df.index)
    bull_choch = pd.Series(False, index=df.index)
    bear_choch = pd.Series(False, index=df.index)
    
    last_ph_idx = -1
    last_pl_idx = -1
    last_ph_val = np.nan
    last_pl_val = np.nan
    
    for i in range(pp, n):
        if pivothigh.iloc[i]:
            last_ph_idx = i
            last_ph_val = ph_value.iloc[i]
            if not np.isnan(major_low.iloc[i]):
                mh_val = major_high.iloc[i] if not np.isnan(major_high.iloc[i]) else last_ph_val
                if last_ph_val > mh_val:
                    major_high.iloc[i] = last_ph_val
                    major_h_idx.iloc[i] = last_ph_idx
                else:
                    major_high.iloc[i] = mh_val
                    major_h_idx.iloc[i] = major_h_idx.iloc[i]
            else:
                major_high.iloc[i] = last_ph_val
                major_h_idx.iloc[i] = last_ph_idx
        
        if pivotlow.iloc[i]:
            last_pl_idx = i
            last_pl_val = pl_value.iloc[i]
            if not np.isnan(major_high.iloc[i]):
                ml_val = major_low.iloc[i] if not np.isnan(major_low.iloc[i]) else last_pl_val
                if last_pl_val < ml_val:
                    major_low.iloc[i] = last_pl_val
                    major_l_idx.iloc[i] = last_pl_idx
                else:
                    major_low.iloc[i] = ml_val
                    major_l_idx.iloc[i] = major_l_idx.iloc[i]
            else:
                major_low.iloc[i] = last_pl_val
                major_l_idx.iloc[i] = last_pl_idx
        
        if not np.isnan(major_high.iloc[i]) and not np.isnan(major_h_idx.iloc[i]):
            mhi = int(major_h_idx.iloc[i])
            if last_pl_idx > mhi and last_pl_val < major_low.iloc[i] - atr.iloc[i]:
                bull_bos.iloc[i] = True
        
        if not np.isnan(major_low.iloc[i]) and not np.isnan(major_l_idx.iloc[i]):
            mli = int(major_l_idx.iloc[i])
            if last_ph_idx > mli and last_ph_val > major_high.iloc[i] + atr.iloc[i]:
                bear_bos.iloc[i] = True
        
        if pivothigh.iloc[i] and last_pl_idx > 0:
            pl_i = last_pl_idx
            if not np.isnan(major_h_idx.iloc[i]):
                mhi = int(major_h_idx.iloc[i])
                if pl_i > mhi and last_ph_val > major_high.iloc[i] + atr.iloc[i] * 0.5 and last_ph_val < major_high.iloc[i] + atr.iloc[i]:
                    bull_choch.iloc[i] = True
        
        if pivotlow.iloc[i] and last_ph_idx > 0:
            ph_i = last_ph_idx
            if not np.isnan(major_l_idx.iloc[i]):
                mli = int(major_l_idx.iloc[i])
                if ph_i > mli and last_pl_val < major_low.iloc[i] - atr.iloc[i] * 0.5 and last_pl_val > major_low.iloc[i] - atr.iloc[i]:
                    bear_choch.iloc[i] = True
    
    # Entry conditions
    long_cond = bull_bos | bull_choch
    short_cond = bear_bos | bear_choch
    
    entries = []
    trade_num = 1
    
    for i in range(pp, n):
        if np.isnan(atr.iloc[i]):
            continue
        
        if long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
        
        if short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
    
    return entries