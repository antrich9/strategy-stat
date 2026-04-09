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
    results = []
    trade_num = 1
    
    df = df.copy()
    df['ts'] = df['time']
    df['datetime'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df['date'] = df['datetime'].dt.date
    
    # Time filter: 07:00-09:59 or 12:00-14:59 UTC
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['total_minutes'] = df['hour'] * 60 + df['minute']
    in_session1 = (df['total_minutes'] >= 420) & (df['total_minutes'] <= 599)  # 07:00-09:59
    in_session2 = (df['total_minutes'] >= 720) & (df['total_minutes'] <= 899)  # 12:00-14:59
    df['in_time_filter'] = in_session1 | in_session2
    
    # Previous day high/low
    daily_agg = df.groupby('date')['high'].max().reset_index()
    daily_agg.columns = ['date', 'day_high']
    daily_low = df.groupby('date')['low'].min().reset_index()
    daily_low.columns = ['date', 'day_low']
    daily_agg = daily_agg.merge(daily_low, on='date')
    
    df_dates = df[['date']].drop_duplicates().sort_values('date').reset_index(drop=True)
    df_dates['prev_date'] = df_dates['date'].shift(1)
    daily_agg = daily_agg.merge(df_dates[['date', 'prev_date']], on='date')
    daily_agg = daily_agg.merge(daily_agg[['date', 'day_high', 'day_low']].rename(columns={'date': 'prev_date', 'day_high': 'prev_day_high', 'day_low': 'prev_day_low'}), on='prev_date', how='left')
    
    df = df.merge(daily_agg[['date', 'prev_day_high', 'prev_day_low']], on='date', how='left')
    
    # New day detection
    df['prev_date_check'] = df['date'].shift(1)
    df['is_new_day'] = df['date'] != df['prev_date_check']
    
    # Sweep flags
    flagpdl = False
    flagpdh = False
    prev_date = None
    
    # OB and FVG conditions
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    
    # OB detection (shifted by 1 in Pine, so using current for index 0)
    is_up_current = close > open_
    is_down_current = close < open_
    is_up_prev = close.shift(1) > open_.shift(1)
    is_down_prev = close.shift(1) < open_.shift(1)
    
    ob_up = is_down_prev & is_up_current & (close > high.shift(1))
    ob_down = is_up_prev & is_down_current & (close < low.shift(1))
    
    # FVG detection (1 bar lookback in Pine uses shift(1), 2 bars shift(2))
    fvg_up = low > high.shift(2)
    fvg_down = high < low.shift(2)
    
    # Combined conditions
    bullish_ob_fvg = ob_up & fvg_up
    bearish_ob_fvg = ob_down & fvg_down
    
    for i in range(len(df)):
        curr_date = df['date'].iloc[i]
        
        if df['is_new_day'].iloc[i]:
            flagpdl = False
            flagpdh = False
            prev_date = curr_date
        
        pdh = df['prev_day_high'].iloc[i]
        pdl = df['prev_day_low'].iloc[i]
        
        if pdh is not None and not pd.isna(pdh) and not flagpdh:
            if close.iloc[i] > pdh:
                flagpdh = True
        
        if pdl is not None and not pd.isna(pdl) and not flagpdl:
            if close.iloc[i] < pdl:
                flagpdl = True
        
        in_tf = df['in_time_filter'].iloc[i]
        
        entry_price = df['close'].iloc[i]
        entry_ts = df['ts'].iloc[i]
        
        if in_tf:
            if flagpdl and bullish_ob_fvg.iloc[i] if i > 0 else False:
                if not (pd.isna(pdl) or pd.isna(pdh)):
                    entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                    results.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': int(entry_ts),
                        'entry_time': entry_time,
                        'entry_price_guess': float(entry_price),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(entry_price),
                        'raw_price_b': float(entry_price)
                    })
                    trade_num += 1
                    flagpdl = False
            
            if flagpdh and bearish_ob_fvg.iloc[i] if i > 0 else False:
                if not (pd.isna(pdl) or pd.isna(pdh)):
                    entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                    results.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': int(entry_ts),
                        'entry_time': entry_time,
                        'entry_price_guess': float(entry_price),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(entry_price),
                        'raw_price_b': float(entry_price)
                    })
                    trade_num += 1
                    flagpdh = False
    
    return results