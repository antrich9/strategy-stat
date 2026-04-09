import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    entries = []
    trade_num = 1
    
    # Calculate Wilder ATR (144 period for FVG filtering)
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_144 = tr.ewm(alpha=1/144, adjust=False).mean()
    
    fvg_th = 0.5
    atr_filter = atr_144 * fvg_th
    
    # Bullish/Bearish gap detection
    bull_g = low > high.shift(1)
    bear_g = high < low.shift(1)
    
    # FVG detection
    bull_fvg_upper = pd.Series(np.nan, index=df.index)
    bull_fvg_lower = pd.Series(np.nan, index=df.index)
    bear_fvg_upper = pd.Series(np.nan, index=df.index)
    bear_fvg_lower = pd.Series(np.nan, index=df.index)
    bull_midpoint = pd.Series(np.nan, index=df.index)
    bear_midpoint = pd.Series(np.nan, index=df.index)
    fvg_last_pct = pd.Series(np.nan, index=df.index)
    fvg_active = pd.Series(False, index=df.index)
    last_fvg_bull = pd.Series(False, index=df.index)
    
    bull_g_prev = False
    bear_g_prev = False
    
    for i in range(2, len(df)):
        b_h = high.iloc[i]
        b_l = low.iloc[i]
        b_c = close.iloc[i]
        b_h_1 = high.iloc[i-1]
        b_h_2 = high.iloc[i-2]
        b_l_1 = low.iloc[i-1]
        b_l_2 = low.iloc[i-2]
        b_c_1 = close.iloc[i-1]
        
        bull_g_val = bull_g.iloc[i]
        bear_g_val = bear_g.iloc[i]
        atr_val = atr_filter.iloc[i]
        
        if pd.isna(atr_val):
            continue
            
        bull = (b_l - b_h_2) > atr_val and b_l > b_h_2 and b_c_1 > b_h_2 and not (bull_g_val or bull_g_prev)
        bear = (b_l_2 - b_h) > atr_val and b_h < b_l_2 and b_c_1 < b_l_2 and not (bear_g_val or bear_g_prev)
        
        if bull:
            bull_fvg_upper.iloc[i] = b_h_2
            bull_fvg_lower.iloc[i] = b_l
            bear_fvg_upper.iloc[i] = np.nan
            bear_fvg_lower.iloc[i] = np.nan
            bull_midpoint.iloc[i] = (b_h_2 + b_l) / 2
            fvg_active.iloc[i] = True
            last_fvg_bull.iloc[i] = True
            upper = b_h_2
            lower = b_l
            fvg_last_pct.iloc[i] = 0.0
        elif bear:
            bear_fvg_upper.iloc[i] = b_h
            bear_fvg_lower.iloc[i] = b_l_2
            bull_fvg_upper.iloc[i] = np.nan
            bull_fvg_lower.iloc[i] = np.nan
            bear_midpoint.iloc[i] = (b_h + b_l_2) / 2
            fvg_active.iloc[i] = True
            last_fvg_bull.iloc[i] = False
            upper = b_h
            lower = b_l_2
            fvg_last_pct.iloc[i] = 0.0
        else:
            fvg_active.iloc[i] = False
            
        if fvg_active.iloc[i]:
            fvg_upper = bull_fvg_upper.iloc[i] if last_fvg_bull.iloc[i] else bear_fvg_upper.iloc[i]
            fvg_lower = bull_fvg_lower.iloc[i] if last_fvg_bull.iloc[i] else bear_fvg_lower.iloc[i]
            if pd.notna(fvg_upper) and pd.notna(fvg_lower):
                pct = (fvg_upper - fvg_lower) / (fvg_upper - fvg_lower)
                fvg_last_pct.iloc[i] = pct
                if last_fvg_bull.iloc[i]:
                    bull_midpoint.iloc[i] = (fvg_upper + fvg_lower) / 2
                else:
                    bear_midpoint.iloc[i] = (fvg_upper + fvg_lower) / 2
        
        bull_g_prev = bull_g_val
        bear_g_prev = bear_g_val
    
    # Previous day high/low calculation
    df_temp = df.copy()
    df_temp['date'] = pd.to_datetime(df_temp['time'], unit='s', utc=True).dt.date
    df_temp['prev_day'] = df_temp['date'].shift(1)
    
    prev_day_high = pd.Series(np.nan, index=df.index)
    prev_day_low = pd.Series(np.nan, index=df.index)
    
    for i in range(1, len(df)):
        if df_temp['date'].iloc[i] != df_temp['prev_day'].iloc[i]:
            for j in range(i-1, -1, -1):
                if df_temp['date'].iloc[j] == df_temp['prev_day'].iloc[i]:
                    prev_day_high.iloc[i] = df['high'].iloc[j]
                    prev_day_low.iloc[i] = df['low'].iloc[j]
                    break
    
    flag_pdh = high > prev_day_high
    flag_pdl = low < prev_day_low
    
    # Time window (London hours)
    times = pd.to_datetime(df['time'], unit='s', utc=True)
    hours = times.dt.hour
    minutes = times.dt.minute
    total_minutes = hours * 60 + minutes
    
    morning_start = 7 * 60 + 45
    morning_end = 9 * 60 + 45
    afternoon_start = 14 * 60 + 45
    afternoon_end = 16 * 60 + 45
    
    in_trading_window = ((total_minutes >= morning_start) & (total_minutes < morning_end)) | \
                        ((total_minutes >= afternoon_start) & (total_minutes < afternoon_end))
    
    # Entry conditions
    bull_entry_cond = (
        fvg_active &
        (fvg_last_pct > 0.01) &
        (fvg_last_pct <= 1) &
        last_fvg_bull &
        in_trading_window &
        flag_pdl &
        bull_midpoint.notna() &
        (low > bull_midpoint)
    )
    
    bear_entry_cond = (
        fvg_active &
        (fvg_last_pct > 0.01) &
        (fvg_last_pct <= 1) &
        ~last_fvg_bull &
        in_trading_window &
        flag_pdh &
        bear_midpoint.notna() &
        (high < bear_midpoint)
    )
    
    for i in range(len(df)):
        if bull_entry_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
        elif bear_entry_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
    
    return entries