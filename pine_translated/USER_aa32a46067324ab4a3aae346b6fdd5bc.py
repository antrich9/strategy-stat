import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    high = df['high']
    low = df['low']
    close = df['close']
    open_price = df['open']
    
    # Parameters from Pine Script
    akk_range = 100
    akk_factor = 6.0
    
    # Wilder ATR calculation
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/akk_range, adjust=False).mean()
    
    # DeltaStop
    delta_stop = atr * akk_factor
    
    # TrStop calculation (requires iterative approach)
    tr_stop = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i == 0:
            tr_stop.iloc[i] = open_price.iloc[i] - delta_stop.iloc[i]
        else:
            prev_ts = tr_stop.iloc[i-1]
            curr_open = open_price.iloc[i]
            prev_open = open_price.iloc[i-1]
            ds = delta_stop.iloc[i]
            
            if pd.isna(prev_ts):
                tr_stop.iloc[i] = curr_open - ds
            elif curr_open == prev_ts:
                tr_stop.iloc[i] = prev_ts
            elif prev_open < prev_ts and curr_open < prev_ts:
                tr_stop.iloc[i] = min(prev_ts, curr_open + ds)
            elif prev_open > prev_ts and curr_open > prev_ts:
                tr_stop.iloc[i] = max(prev_ts, curr_open - ds)
            elif curr_open > prev_ts:
                tr_stop.iloc[i] = curr_open - ds
            else:
                tr_stop.iloc[i] = curr_open + ds
    
    # Basic conditions
    basic_long = close > tr_stop
    basic_short = close < tr_stop
    
    # Cross conditions (crossover logic)
    basic_long_1 = basic_long.shift(1)
    basic_short_1 = basic_short.shift(1)
    
    AKKAM_signals_long_cross = (~basic_long_1) & basic_long
    AKKAM_signals_short_cross = (~basic_short_1) & basic_short
    
    # Final conditions with inverse logic
    use_akkam = True
    inverse_akkam = False
    
    AKKAM_signals_long_final = AKKAM_signals_short_cross if inverse_akkam else AKKAM_signals_long_cross
    AKKAM_signals_short_final = AKKAM_signals_long_cross if inverse_akkam else AKKAM_signals_short_cross
    
    # Entries list
    entries = []
    trade_num = 1
    in_position = False
    position_direction = None
    
    for i in range(len(df)):
        if pd.isna(atr.iloc[i]) or pd.isna(tr_stop.iloc[i]):
            continue
        
        if not in_position:
            if AKKAM_signals_long_final.iloc[i]:
                entry_ts = int(df['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close.iloc[i],
                    'raw_price_b': close.iloc[i]
                })
                trade_num += 1
                in_position = True
                position_direction = 'long'
            elif AKKAM_signals_short_final.iloc[i]:
                entry_ts = int(df['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close.iloc[i],
                    'raw_price_b': close.iloc[i]
                })
                trade_num += 1
                in_position = True
                position_direction = 'short'
        else:
            if position_direction == 'long' and not AKKAM_signals_long_final.iloc[i]:
                in_position = False
                position_direction = None
            elif position_direction == 'short' and not AKKAM_signals_short_final.iloc[i]:
                in_position = False
                position_direction = None
    
    return entries