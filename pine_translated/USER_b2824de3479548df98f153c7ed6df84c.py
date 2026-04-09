import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('datetime', inplace=True)
    
    daily_high = df['high'].resample('D').max()
    daily_low = df['low'].resample('D').min()
    prev_day_high = daily_high.shift(1).reindex(df.index)
    prev_day_low = daily_low.shift(1).reindex(df.index)
    
    prev_day_high_filled = prev_day_high.ffill()
    prev_day_low_filled = prev_day_low.ffill()
    
    flagpdh = (df['close'] > prev_day_high_filled) & (df['close'].shift(1) <= prev_day_high_filled)
    flagpdl = (df['close'] < prev_day_low_filled) & (df['close'].shift(1) >= prev_day_low_filled)
    
    lookback_bars = 12
    threshold = 0.0
    
    high_shifted = df['high'].shift(1)
    low_shifted = df['low'].shift(1)
    high_2_shifted = df['high'].shift(2)
    low_2_shifted = df['low'].shift(2)
    close_1_shifted = df['close'].shift(1)
    
    bull_fvg1 = (df['low'] > high_2_shifted) & (close_1_shifted > high_2_shifted)
    bear_fvg1 = (df['high'] < low_2_shifted) & (close_1_shifted < low_2_shifted)
    
    bull_fvg1_filled = bull_fvg1.fillna(False)
    bear_fvg1_filled = bear_fvg1.fillna(False)
    
    bull_since_rev = (~bull_fvg1_filled)[::-1].cumsum()[::-1]
    bull_since = bull_since_rev.shift(1).fillna(0).astype(int)
    bear_since_rev = (~bear_fvg1_filled)[::-1].cumsum()[::-1]
    bear_since = bear_since_rev.shift(1).fillna(0).astype(int)
    
    bull_cond_1 = bull_fvg1 & (bull_since <= lookback_bars)
    bull_cond_1_filled = bull_cond_1.fillna(False)
    bull_since_cond = bull_since.where(bull_cond_1_filled)
    bull_since_filled = bull_since_cond.fillna(method='ffill').fillna(0)
    
    combined_low_bull = np.where(bull_cond_1_filled, 
                                  np.maximum(high_shifted.fillna(0), high_2_shifted.fillna(0)), 
                                  np.nan)
    combined_high_bull = np.where(bull_cond_1_filled, 
                                   np.minimum(low_shifted_2 := low_2_shifted.fillna(0), df['low'].fillna(0)), 
                                   np.nan)
    bull_range = np.where(bull_cond_1_filled, combined_high_bull - combined_low_bull, np.nan)
    bull_result = bull_cond_1_filled & (bull_range >= threshold)
    
    bear_cond_1 = bear_fvg1 & (bear_since <= lookback_bars)
    bear_cond_1_filled = bear_cond_1.fillna(False)
    bear_since_cond = bear_since.where(bear_cond_1_filled)
    bear_since_filled = bear_since_cond.fillna(method='ffill').fillna(0)
    
    combined_high_bear = np.where(bear_cond_1_filled, 
                                   np.minimum(high_shifted.fillna(0), high_2_shifted.fillna(0)), 
                                   np.nan)
    combined_low_bear = np.where(bear_cond_1_filled, 
                                  np.maximum(low_shifted_2 := low_2_shifted.fillna(0), df['low'].fillna(0)), 
                                  np.nan)
    bear_range = np.where(bear_cond_1_filled, combined_high_bear - combined_low_bear, np.nan)
    bear_result = bear_cond_1_filled & (bear_range >= threshold)
    
    long_entry = flagpdh & bull_result
    short_entry = flagpdl & bear_result
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries