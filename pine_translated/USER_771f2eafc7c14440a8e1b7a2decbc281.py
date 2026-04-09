import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    ts = df['time'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Check trading sessions
    in_session = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        dt = datetime.fromtimestamp(ts[i], tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        if (hour == 3) or (hour == 4 and minute == 0):
            in_session[i] = True
        if (hour == 10) or (hour == 11 and minute == 0):
            in_session[i] = True
    
    # External liquidity (1H highest/lowest over 10)
    external_liquidity_high = pd.Series(high).rolling(10).max()
    external_liquidity_low = pd.Series(low).rolling(10).min()
    
    # Market structure shift (5M: rolling highest/lowest over 10)
    rolling_high_10 = pd.Series(close).rolling(10).max()
    rolling_low_10 = pd.Series(close).rolling(10).min()
    
    structure_shift_up = pd.Series(close).crossover(rolling_high_10)
    structure_shift_down = pd.Series(close).crossunder(rolling_low_10)
    
    # Fair Value Gap
    fvg_up = pd.Series(low[2:] > high[:-2], index=range(2, len(df))) if len(df) >= 3 else pd.Series(dtype=bool)
    fvg_down = pd.Series(high[2:] < low[:-2], index=range(2, len(df))) if len(df) >= 3 else pd.Series(dtype=bool)
    
    # Ensure all conditions are aligned as Series
    fvg_up = fvg_up.reindex(range(len(df)), fill_value=False).astype(bool)
    fvg_down = fvg_down.reindex(range(len(df)), fill_value=False).astype(bool)
    
    # Entry conditions
    long_condition = in_session & structure_shift_up & fvg_up & (close > external_liquidity_high.values)
    short_condition = in_session & structure_shift_down & fvg_down & (close < external_liquidity_low.values)
    
    trade_num = 1
    trades = []
    
    for i in range(len(df)):
        if pd.isna(external_liquidity_high.iloc[i]) or pd.isna(external_liquidity_low.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            raw_price_a = close[i]
            raw_price_b = external_liquidity_high.iloc[i]
            
            entry_time = datetime.fromtimestamp(ts[i], tz=timezone.utc).isoformat()
            trades.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts[i]),
                'entry_time': entry_time,
                'entry_price_guess': close[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': raw_price_a,
                'raw_price_b': raw_price_b
            })
            trade_num += 1
        
        if short_condition.iloc[i]:
            raw_price_a = close[i]
            raw_price_b = external_liquidity_low.iloc[i]
            
            entry_time = datetime.fromtimestamp(ts[i], tz=timezone.utc).isoformat()
            trades.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts[i]),
                'entry_time': entry_time,
                'entry_price_guess': close[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': raw_price_a,
                'raw_price_b': raw_price_b
            })
            trade_num += 1
    
    return trades