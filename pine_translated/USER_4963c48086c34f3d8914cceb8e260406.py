import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    open_vals = df['open']
    high = df['high']
    low = df['low']
    
    is_up = close > open_vals
    is_down = close < open_vals
    
    fvg_up = low > high.shift(2)
    fvg_down = high < low.shift(2)
    
    ob_up = is_down.shift(1) & is_up & (close > high.shift(1))
    ob_down = is_up.shift(1) & is_down & (close < low.shift(1))
    
    stacked_bullish = ob_up.shift(1) & fvg_up
    stacked_bearish = ob_down.shift(1) & fvg_down
    
    day_key = df['time'] // 86400
    daily_agg = df.groupby(day_key).agg({'high': 'max', 'low': 'min'})
    daily_agg = daily_agg.reset_index(drop=True)
    daily_agg['day_key'] = daily_agg.index + 1
    
    prev_day_df = pd.DataFrame({
        'day_key': daily_agg['day_key'] + 1,
        'prev_day_high': daily_agg['high'].values,
        'prev_day_low': daily_agg['low'].values
    })
    
    df_indexed = df.copy()
    df_indexed['day_key'] = day_key
    df_indexed = df_indexed.merge(prev_day_df, on='day_key', how='left')
    
    flagpdh = close > df_indexed['prev_day_high']
    flagpdl = close < df_indexed['prev_day_low']
    
    entries = []
    trade_num = 1
    
    for i in range(2, len(df)):
        if pd.isna(df_indexed['prev_day_high'].iloc[i]) or pd.isna(df_indexed['prev_day_low'].iloc[i]):
            continue
        
        long_cond = stacked_bullish.iloc[i] and flagpdh.iloc[i]
        short_cond = stacked_bearish.iloc[i] and flagpdl.iloc[i]
        
        if long_cond:
            ts = df['time'].iloc[i]
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
        elif short_cond:
            ts = df['time'].iloc[i]
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