import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.sort_values('time').reset_index(drop=True)
    
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    volume = df['volume']
    
    vol_filt = volume.shift(1) > volume.rolling(9).mean() * 1.5
    
    tr1 = high - low
    tr2 = high - close.shift(1)
    tr3 = close.shift(1) - low
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/20, adjust=False).mean() / 1.5
    
    prev_high_2 = high.shift(2)
    prev_low_2 = low.shift(2)
    curr_high = high
    curr_low = low
    
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    
    top_imbalance_size = prev_low_2 - curr_high
    bottom_imbalance_size = curr_low - prev_high_2
    
    bfvg = curr_low > prev_high_2
    sfvg = curr_high < prev_low_2
    
    top_imb_bway = (prev_low_2 <= open_price.shift(1)) & (curr_high >= close.shift(1)) & (close < low.shift(1))
    top_imb_xbway = (prev_low_2 <= open_price.shift(1)) & (curr_high >= close.shift(1)) & (close > low.shift(1))
    bot_imb_bway = (prev_high_2 >= open_price.shift(1)) & (curr_low <= close.shift(1)) & (close > high.shift(1))
    bot_imb_xbway = (prev_high_2 >= open_price.shift(1)) & (curr_low <= close.shift(1)) & (close < high.shift(1))
    
    long_condition = bfvg
    short_condition = sfvg
    
    entries = []
    trade_num = 1
    
    for i in range(2, len(df)):
        curr_ts = int(df['time'].iloc[i])
        
        if long_condition.iloc[i]:
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': curr_ts,
                'entry_time': datetime.fromtimestamp(curr_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            }
            entries.append(entry)
            trade_num += 1
        
        if short_condition.iloc[i]:
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': curr_ts,
                'entry_time': datetime.fromtimestamp(curr_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            }
            entries.append(entry)
            trade_num += 1
    
    return entries