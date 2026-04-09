import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame, auto=True, threshold_per=0.0) -> list:
    df = df.copy()
    df['dt'] = pd.to_datetime(df['time'], unit='s')
    df['date'] = df['dt'].dt.date
    df['new_day'] = df['date'] != df['date'].shift(1)
    df['new_day'].iloc[0] = True
    
    prev_date = df['date'].shift(1)
    pdh = df.groupby(prev_date)['high'].max()
    pdl = df.groupby(prev_date)['low'].min()
    df['pd_high'] = df['date'].map(pdh)
    df['pd_low'] = df['date'].map(pdl)
    
    threshold = (df['high'] - df['low']).cumsum() / df['low'].cumsum() / np.arange(1, len(df) + 1) if auto else pd.Series([threshold_per / 100] * len(df), index=df.index)
    
    bull_fvg = (df['low'] > df['high'].shift(2)) & (df['close'].shift(1) > df['high'].shift(2)) & (((df['low'] - df['high'].shift(2)) / df['high'].shift(2)) > threshold)
    bear_fvg = (df['high'] < df['low'].shift(2)) & (df['close'].shift(1) < df['low'].shift(2)) & (((df['low'].shift(2) - df['high']) / df['high']) > threshold)
    new_fvg_max = np.where(bull_fvg, df['low'], np.where(bear_fvg, df['low'].shift(2), np.nan))
    new_fvg_min = np.where(bull_fvg, df['high'].shift(2), np.where(bear_fvg, df['high'], np.nan))
    new_fvg_t = np.where(bull_fvg | bear_fvg, df['time'].shift(2), np.nan)
    
    entries = []
    trade_num = 1
    consecutive_bull = 0
    consecutive_bear = 0
    max_bull_fvg = np.nan
    min_bull_fvg = np.nan
    max_bear_fvg = np.nan
    min_bear_fvg = np.nan
    last_bull_mid = np.nan
    last_bear_mid = np.nan
    t = np.nan
    swept_high = False
    swept_low = False
    pd_high = np.nan
    pd_low = np.nan
    
    for i in range(len(df)):
        if bull_fvg.iloc[i] and (pd.isna(new_fvg_t.iloc[i]) or new_fvg_t.iloc[i] != t):
            bull_count = 0
            bear_count = 0
            t = new_fvg_t.iloc[i]
            consecutive_bull += 1
            consecutive_bear = 0
            if consecutive_bull >= 2:
                last_bull_mid = (new_fvg_min[i] + new_fvg_max[i]) / 2
        elif pd.notna(min_bull_fvg):
            max_bull_fvg = max(min(df['close'].iloc[i], max_bull_fvg), min_bull_fvg)
        
        if bear_fvg.iloc[i] and (pd.isna(new_fvg_t.iloc[i]) or new_fvg_t.iloc[i] != t):
            bear_count = 0
            bull_count = 0
            t = new_fvg_t.iloc[i]
            consecutive_bear += 1
            consecutive_bull = 0
            if consecutive_bear >= 2:
                last_bear_mid = (new_fvg_min[i] + new_fvg_max[i]) / 2
        elif pd.notna(min_bear_fvg):
            min_bear_fvg = min(max(df['close'].iloc[i], min_bear_fvg), max_bear_fvg)
        
        if df['new_day'].iloc[i]:
            swept_high = False
            swept_low = False
            pd_high = df['high'].iloc[i]
            pd_low = df['low'].iloc[i]
        else:
            if not swept_high and df['high'].iloc[i] > pd_high:
                swept_high = True
            if not swept_low and df['low'].iloc[i] < pd_low:
                swept_low = True
        
        if pd.isna(df['pd_high'].iloc[i]) or pd.isna(df['pd_low'].iloc[i]):
            continue
        if bull_fvg.iloc[i] and pd.isna(new_fvg_max[i]):
            continue
        if bear_fvg.iloc[i] and pd.isna(new_fvg_max[i]):
            continue
        
        if swept_low and bull_fvg.iloc[i] and consecutive_bull >= 2:
            ts = int(df['time'].iloc[i])
            entries.append({'trade_num': trade_num, 'direction': 'long', 'entry_ts': ts, 'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(), 'entry_price_guess': float(df['close'].iloc[i]), 'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0, 'raw_price_a': float(df['close'].iloc[i]), 'raw_price_b': float(df['close'].iloc[i])})
            trade_num += 1
        if swept_high and bear_fvg.iloc[i] and consecutive_bear >= 2:
            ts = int(df['time'].iloc[i])
            entries.append({'trade_num': trade_num, 'direction': 'short', 'entry_ts': ts, 'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(), 'entry_price_guess': float(df['close'].iloc[i]), 'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0, 'raw_price_a': float(df['close'].iloc[i]), 'raw_price_b': float(df['close'].iloc[i])})
            trade_num += 1
    
    return entries