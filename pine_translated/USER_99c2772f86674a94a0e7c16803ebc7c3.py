import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

def generate_entries(df: pd.DataFrame) -> list:
    est_offset = timedelta(hours=-5)
    est_tz = timezone(est_offset)
    
    dt = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert(est_tz)
    df = df.copy()
    df['dt'] = dt
    df['est_hour'] = dt.dt.hour
    df['est_minute'] = (dt.dt.hour * 60 + dt.dt.minute) % (24 * 60)
    
    asia_sess = df['est_hour'] >= 18
    london_sess = df['est_hour'] < 6
    sess_6_730 = (df['est_minute'] >= 360) & (df['est_minute'] < 450)
    fvg_time = df['est_hour'] == 10
    
    asia_high_vals = np.full(len(df), np.nan)
    asia_low_vals = np.full(len(df), np.nan)
    london_high_vals = np.full(len(df), np.nan)
    london_low_vals = np.full(len(df), np.nan)
    hi6_730_vals = np.full(len(df), np.nan)
    lo6_730_vals = np.full(len(df), np.nan)
    
    new_day = dt.diff().dt.days > 0
    new_day.iloc[0] = True
    
    asia_high, asia_low = np.nan, np.nan
    london_high, london_low = np.nan, np.nan
    hi6_730, lo6_730 = np.nan, np.nan
    entered_today = False
    bias = ""
    
    for i in range(len(df)):
        if new_day.iloc[i]:
            entered_today = False
            asia_high, asia_low = np.nan, np.nan
            london_high, london_low = np.nan, np.nan
            hi6_730, lo6_730 = np.nan, np.nan
            bias = ""
        
        if asia_sess.iloc[i]:
            if np.isnan(asia_high):
                asia_high = df['high'].iloc[i]
            else:
                asia_high = max(asia_high, df['high'].iloc[i])
            if np.isnan(asia_low):
                asia_low = df['low'].iloc[i]
            else:
                asia_low = min(asia_low, df['low'].iloc[i])
            asia_high_vals[i] = asia_high
            asia_low_vals[i] = asia_low
        
        if london_sess.iloc[i]:
            if np.isnan(london_high):
                london_high = df['high'].iloc[i]
            else:
                london_high = max(london_high, df['high'].iloc[i])
            if np.isnan(london_low):
                london_low = df['low'].iloc[i]
            else:
                london_low = min(london_low, df['low'].iloc[i])
            london_high_vals[i] = london_high
            london_low_vals[i] = london_low
        
        if sess_6_730.iloc[i]:
            if np.isnan(hi6_730):
                hi6_730 = df['high'].iloc[i]
            else:
                hi6_730 = max(hi6_730, df['high'].iloc[i])
            if np.isnan(lo6_730):
                lo6_730 = df['low'].iloc[i]
            else:
                lo6_730 = min(lo6_730, df['low'].iloc[i])
            hi6_730_vals[i] = hi6_730
            lo6_730_vals[i] = lo6_730
        
        if df['est_hour'].iloc[i] == 9 and dt.dt.minute.iloc[i] == 30:
            asia_high, asia_low = np.nan, np.nan
            london_high, london_low = np.nan, np.nan
            hi6_730, lo6_730 = np.nan, np.nan
    
    df['asia_high'] = pd.Series(asia_high_vals, index=df.index)
    df['asia_low'] = pd.Series(asia_low_vals, index=df.index)
    df['london_high'] = pd.Series(london_high_vals, index=df.index)
    df['london_low'] = pd.Series(london_low_vals, index=df.index)
    df['hi6_730'] = pd.Series(hi6_730_vals, index=df.index)
    df['lo6_730'] = pd.Series(lo6_730_vals, index=df.index)
    
    swept_high = ((~df['asia_high'].isna() & (df['high'] >= df['asia_high'])) | 
                  (~df['london_high'].isna() & (df['high'] >= df['london_high'])) |
                  (~df['hi6_730'].isna() & (df['high'] >= df['hi6_730'])))
    swept_low = ((~df['asia_low'].isna() & (df['low'] <= df['asia_low'])) |
                 (~df['london_low'].isna() & (df['low'] <= df['london_low'])) |
                 (~df['lo6_730'].isna() & (df['low'] <= df['lo6_730'])))
    
    bias_arr = pd.Series(index=df.index, dtype='object')
    bias_val = ""
    for i in df.index:
        if swept_high.iloc[i]:
            bias_val = "short"
        if swept_low.iloc[i]:
            bias_val = "long"
        bias_arr.iloc[i] = bias_val
    
    fvg_bull = df['low'] > df['high'].shift(2)
    fvg_bear = df['high'] < df['low'].shift(2)
    
    can_trade_arr = pd.Series(False, index=df.index)
    can_trade_val = False
    for i in df.index:
        if new_day.iloc[i]:
            can_trade_val = False
        if not can_trade_val and fvg_time.iloc[i] and bias_arr.iloc[i] != "":
            can_trade_val = True
        can_trade_arr.iloc[i] = can_trade_val
    
    entries = []
    trade_num = 1
    entered_today = False
    
    for i in df.index:
        if new_day.iloc[i]:
            entered_today = False
        if can_trade_arr.iloc[i] and bias_arr.iloc[i] == "long" and fvg_bull.iloc[i] and not entered_today:
            entry_price = df['close'].iloc[i]
            raw_price_a = entry_price
            raw_price_b = entry_price
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': raw_price_a,
                'raw_price_b': raw_price_b
            })
            trade_num += 1
            entered_today = True
        if can_trade_arr.iloc[i] and bias_arr.iloc[i] == "short" and fvg_bear.iloc[i] and not entered_today:
            entry_price = df['close'].iloc[i]
            raw_price_a = entry_price
            raw_price_b = entry_price
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': raw_price_a,
                'raw_price_b': raw_price_b
            })
            trade_num += 1
            entered_today = True
    
    return entries