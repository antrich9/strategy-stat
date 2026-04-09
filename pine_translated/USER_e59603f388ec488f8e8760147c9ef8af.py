import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Constants from Pine Script
    fib_level = 71
    stop_loss_pct = 1.5 # not used
    take_profit_pct = 3.0 # not used
    
    entries = []
    trade_num = 1
    
    # Higher Time Frame Direction (assuming df is the higher timeframe data)
    higher_tf_close = df['close']
    # higher_tf_direction = 1 if close > close[1] else -1
    higher_tf_direction = np.where(higher_tf_close > higher_tf_close.shift(1), 1, -1)
    higher_tf_direction_series = pd.Series(higher_tf_direction, index=df.index)
    
    # Break of Structure
    # close > ta.highest(close, 20)[1] (assuming previous highest based on comment "previous 20")
    # If literal: close > close.rolling(20).max() -> impossible.
    # If shifted: close > close.rolling(20).max().shift(1)
    bos = df['close'] > df['close'].rolling(20).max().shift(1)
    
    # Fibonacci Retracement Level
    # fib_level = src * (1 - 71/100)
    retracement_level = df['close'] * (1 - fib_level / 100)
    
    # Conditions
    # long_condition = (higher_tf_direction == 1) and is_break_of_structure and close <= retracement_level
    long_cond = (higher_tf_direction_series == 1) & (bos) & (df['close'] <= retracement_level)
    # short_condition = (higher_tf_direction == -1) and is_break_of_structure and close >= retracement_level
    short_cond = (higher_tf_direction_series == -1) & (bos) & (df['close'] >= retracement_level)
    
    for i in range(len(df)):
        # Check for NaN in indicators at bar i
        # bos.iloc[i] can be NaN/True/False. If NaN, skip.
        if pd.isna(bos.iloc[i]):
            continue
        # retracement_level should be fine but just in case
        if pd.isna(retracement_level.iloc[i]):
            continue
            
        ts = df['time'].iloc[i]
        entry_price = df['close'].iloc[i]
        
        if long_cond.iloc[i]:
            entries.append({
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
            
        if short_cond.iloc[i]:
            entries.append({
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
            
    return entries