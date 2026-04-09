import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame,
                     atr_length: int = 14,
                     length_md: int = 10,
                     ci_length: int = 14,
                     trade_direction: str = 'Both',
                     lookback: int = 20,
                     ret_since: int = 2,
                     ret_valid_limit: int = 2) -> list:
    """
    Entry logic for Choppiness Index ratio Breakout and Retest Strategy with ATR SL/TP
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    lookback = int(lookback)
    ret_since = int(ret_since)
    ret_valid_limit = int(ret_valid_limit)
    
    # Calculate ATR using Wilder smoothing
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = pd.Series(index=df.index, dtype=float)
    if len(tr) >= atr_length:
        atr.iloc[atr_length - 1] = tr.iloc[:atr_length].mean()
        for i in range(atr_length, len(tr)):
            atr.iloc[i] = (atr.iloc[i - 1] * (atr_length - 1) + tr.iloc[i]) / atr_length
    atr = atr.fillna(0)
    
    # Choppiness Index calculation
    atr_sum = tr.rolling(window=ci_length).mean()
    high_range = high.rolling(window=ci_length).max()
    low_range = low.rolling(window=ci_length).min()
    high_low_range = high_range - low_range
    ci = 100 * np.log10(atr_sum / high_low_range) / np.log10(ci_length)
    
    # McGinley Dynamic
    md = pd.Series(index=df.index, dtype=float)
    md.iloc[0] = close.iloc[0]
    for i in range(1, len(close)):
        if md.iloc[i - 1] != 0:
            md.iloc[i] = md.iloc[i - 1] + (close.iloc[i] - md.iloc[i - 1]) / (length_md * ((close.iloc[i] / md.iloc[i - 1]) ** 4))
        else:
            md.iloc[i] = close.iloc[i]
    
    # Pivot detection
    def get_pivot_low(idx, lb):
        if idx < lb:
            return np.nan
        window = low.iloc[idx - lb:idx + lb + 1]
        min_val = window.min()
        if min_val == low.iloc[idx] and idx == window.idxmin():
            return low.iloc[idx]
        return np.nan
    
    def get_pivot_high(idx, lb):
        if idx < lb:
            return np.nan
        window = high.iloc[idx - lb:idx + lb + 1]
        max_val = window.max()
        if max_val == high.iloc[idx] and idx == window.idxmax():
            return high.iloc[idx]
        return np.nan
    
    # Initialize state variables
    sBreak = False
    rBreak = False
    sBreak_bar = -1
    rBreak_bar = -1
    retOccurred_s = False
    retOccurred_r = False
    sRet_event_bar = -1
    rRet_event_bar = -1
    sRet_event_val = np.nan
    rRet_event_val = np.nan
    
    trade_num = 1
    entries = []
    
    min_len = lookback + 1
    
    for i in range(min_len, len(df)):
        if pd.isna(ci.iloc[i]):
            continue
        
        # Current support box
        pl = get_pivot_low(i, lookback)
        ph = get_pivot_high(i, lookback)
        
        sTop = np.nan
        sBot = np.nan
        rTop = np.nan
        rBot = np.nan
        
        if not pd.isna(pl):
            idx_l = i - lookback
            s_yLoc = low.iloc[idx_l + 1] if low.iloc[idx_l + 1] > low.iloc[idx_l - 1] else low.iloc[idx_l - 1]
            sTop = pl
            sBot = s_yLoc
        
        if not pd.isna(ph):
            idx_h = i - lookback
            r_yLoc = high.iloc[idx_h + 1] if high.iloc[idx_h + 1] > high.iloc[idx_h - 1] else high.iloc[idx_h - 1]
            rTop = r_yLoc
            rBot = ph
        
        # Support breakout detection
        if not pd.isna(sBot):
            if i > 0 and close.iloc[i] < sBot and close.iloc[i - 1] >= sBot:
                if not sBreak:
                    sBreak = True
                    sBreak_bar = i
        
        # Resistance breakout detection
        if not pd.isna(rTop):
            if i > 0 and close.iloc[i] > rTop and close.iloc[i - 1] <= rTop:
                if not rBreak:
                    rBreak = True
                    rBreak_bar = i
        
        # Retest event detection
        if sBreak and not retOccurred_s:
            bars_since_break = i - sBreak_bar
            if bars_since_break > ret_since:
                s1 = high.iloc[i] >= sTop and close.iloc[i] <= sBot
                s2 = high.iloc[i] >= sTop and close.iloc[i] >= sBot and close.iloc[i] <= sTop
                s3 = high.iloc[i] >= sBot and high.iloc[i] <= sTop
                s4 = high.iloc[i] >= sBot and high.iloc[i] <= sTop and close.iloc[i] < sBot
                
                if s1 or s2 or s3 or s4:
                    retOccurred_s = True
                    sRet_event_bar = i
                    sRet_event_val = high.iloc[i]
        
        if rBreak and not retOccurred_r:
            bars_since_break = i - rBreak_bar
            if bars_since_break > ret_since:
                r1 = low.iloc[i] <= rBot and close.iloc[i] >= rTop
                r2 = low.iloc[i] <= rBot and close.iloc[i] <= rTop and close.iloc[i] >= rBot
                r3 = low.iloc[i] <= rTop and low.iloc[i] >= rBot
                r4 = low.iloc[i] <= rTop and low.iloc[i] >= rBot and close.iloc[i] > rTop
                
                if r1 or r2 or r3 or r4:
                    retOccurred_r = True
                    rRet_event_bar = i
                    rRet_event_val = low.iloc[i]
        
        # Entry condition checks
        if sBreak and retOccurred_s:
            bars_since_ret = i - sRet_event_bar
            retConditions = close.iloc[i] <= sRet_event_val
            
            if bars_since_ret > 0 and bars_since_ret <= ret_valid_limit and retConditions:
                if trade_direction in ['Long', 'Both']:
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': int(df['time'].iloc[i]),
                        'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                        'entry_price_guess': float(close.iloc[i]),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(close.iloc[i]),
                        'raw_price_b': float(close.iloc[i])
                    })
                    trade_num += 1
                
                sBreak = False
                retOccurred_s = False
                sRet_event_bar = -1
                sRet_event_val = np.nan
        
        if rBreak and retOccurred_r:
            bars_since_ret = i - rRet_event_bar
            retConditions = close.iloc[i] >= rRet_event_val
            
            if bars_since_ret > 0 and bars_since_ret <= ret_valid_limit and retConditions:
                if trade_direction in ['Short', 'Both']:
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': int(df['time'].iloc[i]),
                        'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                        'entry_price_guess': float(close.iloc[i]),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(close.iloc[i]),
                        'raw_price_b': float(close.iloc[i])
                    })
                    trade_num += 1
                
                rBreak = False
                retOccurred_r = False
                rRet_event_bar = -1
                rRet_event_val = np.nan
    
    return entries