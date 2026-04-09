import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['time_dt'].dt.date
    
    # Previous day high/low calculation
    daily_agg = df.groupby('date').agg({'high': 'max', 'low': 'min'}).reset_index()
    daily_agg['date_prev'] = daily_agg['date'].shift(1)
    daily_agg = daily_agg[['date_prev', 'high', 'low']].rename(columns={'date_prev': 'date', 'high': 'pdh', 'low': 'pdl'})
    df = df.merge(daily_agg, on='date', how='left')
    
    # Sweep detection
    df['sweptHigh'] = False
    df['sweptLow'] = False
    
    for i in range(1, len(df)):
        if df['date'].iloc[i] != df['date'].iloc[i-1]:
            df.loc[df.index[i], 'sweptHigh'] = False
            df.loc[df.index[i], 'sweptLow'] = False
        else:
            df.loc[df.index[i], 'sweptHigh'] = df.loc[df.index[i-1], 'sweptHigh']
            df.loc[df.index[i], 'sweptLow'] = df.loc[df.index[i-1], 'sweptLow']
        
        if df['high'].iloc[i] > df['pdh'].iloc[i] and not df.loc[df.index[i-1], 'sweptHigh']:
            df.loc[df.index[i], 'sweptHigh'] = True
        if df['low'].iloc[i] < df['pdl'].iloc[i] and not df.loc[df.index[i-1], 'sweptLow']:
            df.loc[df.index[i], 'sweptLow'] = True
    
    df['sweepHighNow'] = (~df['sweptHigh']) & (df['high'] > df['pdh']) & df['pdh'].notna()
    df['sweepLowNow'] = (~df['sweptLow']) & (df['low'] < df['pdl']) & df['pdl'].notna()
    
    # FVG detection
    threshold = 0.005
    
    high_shifted2 = df['high'].shift(2)
    low_shifted2 = df['low'].shift(2)
    close_shifted1 = df['close'].shift(1)
    
    df['bull_fvg'] = (df['low'] > high_shifted2) & (close_shifted1 > high_shifted2) & ((df['low'] - high_shifted2) / high_shifted2 > threshold)
    df['bear_fvg'] = (df['high'] < low_shifted2) & (close_shifted1 < low_shifted2) & ((low_shifted2 - df['high']) / df['high'] > threshold)
    
    bull_fvg = df['bull_fvg']
    bear_fvg = df['bear_fvg']
    
    # Consecutive tracking
    consecutive_bull = 0
    consecutive_bear = 0
    last_bull_fvg_max = np.nan
    last_bull_fvg_min = np.nan
    last_bear_fvg_max = np.nan
    last_bear_fvg_min = np.nan
    entries = []
    trade_num = 1
    
    for i in range(2, len(df)):
        prev_date = df['date'].iloc[i-1]
        curr_date = df['date'].iloc[i]
        
        if curr_date != prev_date:
            consecutive_bull = 0
            consecutive_bear = 0
        
        if bull_fvg.iloc[i]:
            new_fvg_min = df['low'].iloc[i]
            new_fvg_max = high_shifted2.iloc[i]
            
            if new_fvg_min != last_bull_fvg_min or new_fvg_max != last_bull_fvg_max:
                consecutive_bull = 1
                consecutive_bear = 0
                last_bull_fvg_min = new_fvg_min
                last_bull_fvg_max = new_fvg_max
            else:
                consecutive_bull += 1
            
            if consecutive_bull >= 2:
                midpoint = (new_fvg_min + new_fvg_max) / 2
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': midpoint,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': midpoint,
                    'raw_price_b': midpoint
                })
                trade_num += 1
                consecutive_bull = 0
        elif bear_fvg.iloc[i]:
            new_fvg_min = low_shifted2.iloc[i]
            new_fvg_max = df['high'].iloc[i]
            
            if new_fvg_min != last_bear_fvg_min or new_fvg_max != last_bear_fvg_max:
                consecutive_bear = 1
                consecutive_bull = 0
                last_bear_fvg_min = new_fvg_min
                last_bear_fvg_max = new_fvg_max
            else:
                consecutive_bear += 1
            
            if consecutive_bear >= 2:
                midpoint = (new_fvg_min + new_fvg_max) / 2
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': midpoint,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': midpoint,
                    'raw_price_b': midpoint
                })
                trade_num += 1
                consecutive_bear = 0
        
        if df['sweptHigh'].iloc[i] and df['sweptLow'].iloc[i]:
            consecutive_bull = 0
            consecutive_bear = 0
    
    return entries