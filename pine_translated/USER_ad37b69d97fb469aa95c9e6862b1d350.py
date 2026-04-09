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
    df = df.copy().reset_index(drop=True)
    
    dt = pd.to_datetime(df['time'], unit='s', utc=True)
    hours = dt.dt.hour
    
    asian_start_hour = 23
    asian_end_hour = 6
    
    in_asian = ((asian_start_hour > asian_end_hour) & 
                ((hours >= asian_start_hour) | (hours < asian_end_hour))) | \
               ((asian_start_hour <= asian_end_hour) & 
                (hours >= asian_start_hour) & (hours < asian_end_hour))
    
    in_asian_prev = in_asian.shift(1).fillna(False)
    asian_ended = (~in_asian) & in_asian_prev
    
    df['in_asian'] = in_asian
    df['asian_ended'] = asian_ended
    
    asian_session_id = (in_asian).cumsum()
    asian_session_id_adj = asian_session_id - 1
    df['session_group'] = asian_session_id_adj
    
    asian_high_vals = df.loc[in_asian].groupby(df['session_group'])['high'].max()
    asian_low_vals = df.loc[in_asian].groupby(df['session_group'])['low'].min()
    
    asian_high_map = df['session_group'].map(asian_high_vals).ffill()
    asian_low_map = df['session_group'].map(asian_low_vals).ffill()
    
    df['asian_high'] = asian_high_map.where(in_asian.shift(1).fillna(False), np.nan).ffill().ffill()
    df['asian_low'] = asian_low_map.where(in_asian.shift(1).fillna(False), np.nan).ffill().ffill()
    
    sweep_high = (df['high'] > df['asian_high']) & (df['high'].shift(1) <= df['asian_high'])
    sweep_low = (df['low'] < df['asian_low']) & (df['low'].shift(1) >= df['asian_low'])
    
    df['swept_high'] = sweep_high.cumsum()
    df['swept_low'] = sweep_low.cumsum()
    
    london_start = dt.dt.replace(hour=14, minute=45, second=0, microsecond=0)
    london_end = dt.dt.replace(hour=16, minute=45, second=0, microsecond=0)
    in_window = (dt >= london_start) & (dt < london_end)
    
    long_cond = sweep_high & (~df['swept_high'].shift(1).fillna(0).astype(bool)) & in_window
    short_cond = sweep_low & (~df['swept_low'].shift(1).fillna(0).astype(bool)) & in_window
    
    long_cond = long_cond & df['asian_high'].notna()
    short_cond = short_cond & df['asian_low'].notna()
    
    trades = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = df['close'].iloc[i]
            trades.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = df['close'].iloc[i]
            trades.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return trades