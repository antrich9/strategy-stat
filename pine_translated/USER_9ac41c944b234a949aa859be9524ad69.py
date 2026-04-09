import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Parameters from strategy
    left = 20
    right = 15
    nPiv = 4
    atrLen = 30
    mult = 0.5
    per = 5.0
    max_boxes = 10
    src_option = "HA"
    
    # Detection flags
    dhighs = True
    dlows = True
    detectBO = False
    detectBD = False
    breakUp = False
    breakDn = False
    falseBull = False
    falseBear = False
    
    # Calculate Heikin Ashi if needed
    if src_option == "HA":
        ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha_open = pd.Series(index=df.index, dtype=float)
        ha_open.iloc[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
        for i in range(1, len(df)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
        src_series = ha_close
    else:
        src_series = df['close']
    
    # Calculate pivots
    pivot_high = pd.Series(index=df.index, dtype=float)
    pivot_low = pd.Series(index=df.index, dtype=float)
    
    for i in range(right, len(df) - left):
        if dhighs:
            high_slice = src_series.iloc[i-right:i]
            pivot_high.iloc[i] = high_slice.max()
        if dlows:
            low_slice = src_series.iloc[i-right:i]
            pivot_low.iloc[i] = low_slice.min()
    
    # Wilder ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    
    atr = pd.Series(index=df.index, dtype=float)
    atr.iloc[atrLen - 1] = tr.iloc[:atrLen].mean()
    for i in range(atrLen, len(df)):
        atr.iloc[i] = (atr.iloc[i-1] * (atrLen - 1) + tr.iloc[i]) / atrLen
    
    # Calculate zones for each pivot level
    for level in range(nPiv):
        zone_attr = f'zone_high_{level}'
        zone_low_attr = f'zone_low_{level}'
    
    # Build arrays of pivot highs and lows with timestamps
    pivot_highs = []
    pivot_lows = []
    for i in range(len(df)):
        if not pd.isna(pivot_high.iloc[i]):
            pivot_highs.append((df['time'].iloc[i], pivot_high.iloc[i]))
        if not pd.isna(pivot_low.iloc[i]):
            pivot_lows.append((df['time'].iloc[i], pivot_low.iloc[i]))
    
    # Sort by time
    pivot_highs.sort(key=lambda x: x[0])
    pivot_lows.sort(key=lambda x: x[0])
    
    # Keep only nPiv most recent
    if len(pivot_highs) > nPiv:
        pivot_highs = pivot_highs[-nPiv:]
    if len(pivot_lows) > nPiv:
        pivot_lows = pivot_lows[-nPiv:]
    
    # Calculate resistance and support levels with zone width
    zone_width = atr * mult
    
    resistance_levels = []
    support_levels = []
    
    for ph_time, ph_val in pivot_highs:
        resistance_levels.append((ph_time, ph_val + zone_width.iloc[np.searchsorted(df['time'].values, ph_time)] if len(df[df['time'] <= ph_time]) > 0 else ph_val + atr.iloc[0] * mult))
    
    for pl_time, pl_val in pivot_lows:
        support_levels.append((pl_time, pl_val - zone_width.iloc[np.searchsorted(df['time'].values, pl_time)] if len(df[df['time'] <= pl_time]) > 0 else pl_val - atr.iloc[0] * mult))
    
    # Extract current resistance and support levels
    current_resistance = max([r[1] for r in resistance_levels]) if resistance_levels else df['high'].max()
    current_support = min([s[1] for s in support_levels]) if support_levels else df['low'].min()
    
    # Find max pivot values for breakout/breakdown detection
    max_pivot_high = max([p[1] for p in pivot_highs]) if pivot_highs else 0
    min_pivot_low = min([p[1] for p in pivot_lows]) if pivot_lows else float('inf')
    
    # Generate entries
    entries = []
    trade_num = 1
    
    in_long = False
    in_short = False
    
    for i in range(1, len(df)):
        close = df['close'].iloc[i]
        prev_close = df['close'].iloc[i-1]
        ts = int(df['time'].iloc[i])
        
        # Skip if indicators not ready
        if i < atrLen + right:
            continue
        
        current_atr = atr.iloc[i]
        zone = current_atr * mult
        
        # Calculate current zones based on recent pivots
        recent_highs = [p[1] for p in pivot_highs if p[0] <= df['time'].iloc[i]][-nPiv:] if pivot_highs else []
        recent_lows = [p[1] for p in pivot_lows if p[0] <= df['time'].iloc[i]][-nPiv:] if pivot_lows else []
        
        if recent_highs:
            local_resistance = max(recent_highs) + zone
        else:
            local_resistance = close + zone
        
        if recent_lows:
            local_support = min(recent_lows) - zone
        else:
            local_support = close - zone
        
        # Find max high and min low of all pivots
        all_pivot_highs = [p[1] for p in pivot_highs if p[0] <= df['time'].iloc[i]]
        all_pivot_lows = [p[1] for p in pivot_lows if p[0] <= df['time'].iloc[i]]
        
        max_all_pivot = max(all_pivot_highs) if all_pivot_highs else 0
        min_all_pivot = min(all_pivot_lows) if all_pivot_lows else float('inf')
        
        # Detect breakout (close above all pivot highs)
        breakout = close > max_all_pivot if all_pivot_highs else False
        prev_breakout = prev_close > max(all_pivot_highs[:-1]) if len(all_pivot_highs) > 1 else False
        
        # Detect breakdown (close below all pivot lows)
        breakdown = close < min_all_pivot if all_pivot_lows else False
        prev_breakdown = prev_close < min(all_pivot_lows[:-1]) if len(all_pivot_lows) > 1 else False
        
        # Break above resistance
        break_above_res = close > local_resistance and prev_close <= local_resistance
        
        # Break below support
        break_below_sup = close < local_support and prev_close >= local_support
        
        # Long entries
        if detectBO and breakout and not prev_breakout and not in_long:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close,
                'raw_price_b': close
            })
            trade_num += 1
            in_long = True
            in_short = False
        
        if breakUp and break_above_res and not in_long:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close,
                'raw_price_b': close
            })
            trade_num += 1
            in_long = True
            in_short = False
        
        # Short entries
        if detectBD and breakdown and not prev_breakdown and not in_short:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close,
                'raw_price_b': close
            })
            trade_num += 1
            in_short = True
            in_long = False
        
        if breakDn and break_below_sup and not in_short:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close,
                'raw_price_b': close
            })
            trade_num += 1
            in_short = True
            in_long = False
    
    return entries