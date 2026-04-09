import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['time_decimal'] = df['hour'] + df['minute'] / 60.0
    
    in_morning_window = (df['time_decimal'] >= 7.75) & (df['time_decimal'] <= 9.75)
    in_afternoon_window = (df['time_decimal'] >= 14.75) & (df['time_decimal'] <= 16.75)
    is_within_time_window = in_morning_window | in_afternoon_window
    
    is_up = df['close'] > df['open']
    is_down = df['close'] < df['open']
    is_ob_up = is_down.shift(1) & is_up & (df['close'] > df['high'].shift(1))
    is_ob_down = is_up.shift(1) & is_down & (df['close'] < df['low'].shift(1))
    
    is_fvg_up = df['low'] > df['high'].shift(2)
    is_fvg_down = df['high'] < df['low'].shift(2)
    
    bull_stack = is_ob_up & is_fvg_up
    bear_stack = is_ob_down & is_fvg_down
    
    bfvg = is_fvg_up
    sfvg = is_fvg_down
    
    vol_ma = df['volume'].rolling(9).mean()
    vol_filt = df['volume'] > vol_ma * 1.5
    
    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = tr.rolling(20).mean() / 1.5
    atrfilt = ((df['low'] - df['high'].shift(2)) > atr) | ((df['low'].shift(2) - df['high']) > atr)
    
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    bull_cond = bull_stack.shift(1) | bfvg.shift(1)
    bear_cond = bear_stack.shift(1) | sfvg.shift(1)
    
    bull_entry = bull_cond & is_within_time_window & vol_filt & atrfilt & locfiltb
    bear_entry = bear_cond & is_within_time_window & vol_filt & atrfilt & locfilts
    
    entries = []
    trade_num = 0
    
    for i in range(1, len(df)):
        if bull_entry.iloc[i]:
            trade_num += 1
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
        elif bear_entry.iloc[i]:
            trade_num += 1
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
    
    return entries