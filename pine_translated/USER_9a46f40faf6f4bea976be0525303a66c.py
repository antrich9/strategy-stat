import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    pivotLen = 14
    atrPeriod = 14
    fvgWaitBars = 10
    fvgMinTicks = 3

    # Calculate Wilder ATR manually
    high = df['high']
    low = df['low']
    close = df['close']
    
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    
    atr = pd.Series(index=df.index, dtype=float)
    atr.iloc[pivotLen] = tr.iloc[1:pivotLen+1].mean()
    for i in range(pivotLen + 1, len(df)):
        atr.iloc[i] = (atr.iloc[i-1] * (atrPeriod - 1) + tr.iloc[i]) / atrPeriod
    
    # Calculate pivothigh and pivotlow (shifted to align at bar_index - pivotLen)
    ph = high.shift(pivotLen).rolling(pivotLen+1).max()
    pl = low.shift(pivotLen).rolling(pivotLen+1).min()
    
    # Track levels
    levels_h = []  # Bear zones (resistance at pivot highs)
    levels_l = []  # Bull zones (support at pivot lows)
    
    # Track sweeps
    last_bear_sweep_bar = None
    last_bear_sweep_price = np.nan
    last_bull_sweep_bar = None
    last_bull_sweep_price = np.nan
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        # Check for new pivot highs (bear zones)
        if not pd.isna(ph.iloc[i]):
            pivot_bar = i - pivotLen
            levels_h.append({
                'price': ph.iloc[i],
                'created_bar': pivot_bar,
                'active': True,
                'swept': False
            })
        
        # Check for new pivot lows (bull zones)
        if not pd.isna(pl.iloc[i]):
            pivot_bar = i - pivotLen
            levels_l.append({
                'price': pl.iloc[i],
                'created_bar': pivot_bar,
                'active': True,
                'swept': False
            })
        
        # Process Bear Zones (high sweeps)
        for lvl in levels_h:
            if lvl['active']:
                # BEAR SWEEP: Price wicks above but closes below
                if high.iloc[i] > lvl['price'] and close.iloc[i] < lvl['price'] and not lvl['swept']:
                    lvl['swept'] = True
                    last_bear_sweep_bar = i
                    last_bear_sweep_price = lvl['price']
        
        # Process Bull Zones (low sweeps)
        for lvl in levels_l:
            if lvl['active']:
                # BULL SWEEP: Price wicks below but closes above
                if low.iloc[i] < lvl['price'] and close.iloc[i] > lvl['price'] and not lvl['swept']:
                    lvl['swept'] = True
                    last_bull_sweep_bar = i
                    last_bull_sweep_price = lvl['price']
        
        # FVG detection
        if i >= 2:
            # Bullish FVG (gap up unfilled)
            bullish_fvg = low.iloc[i] > high.iloc[i-2]
            bullish_fvg_size = low.iloc[i] - high.iloc[i-2] if bullish_fvg else 0
            # Min tick size - using very small value as placeholder
            min_tick = 0.01
            bullish_fvg_valid = bullish_fvg and (bullish_fvg_size / min_tick >= fvgMinTicks)
            
            # Bearish FVG (gap down unfilled)
            bearish_fvg = high.iloc[i] < low.iloc[i-2]
            bearish_fvg_size = low.iloc[i-2] - high.iloc[i] if bearish_fvg else 0
            bearish_fvg_valid = bearish_fvg and (bearish_fvg_size / min_tick >= fvgMinTicks)
        else:
            bullish_fvg_valid = False
            bearish_fvg_valid = False
        
        # Check if we need to clear old sweeps
        bars_since_bear_sweep = 999 if last_bear_sweep_bar is None else (i - last_bear_sweep_bar)
        bars_since_bull_sweep = 999 if last_bull_sweep_bar is None else (i - last_bull_sweep_bar)
        
        if bars_since_bear_sweep > fvgWaitBars:
            last_bear_sweep_bar = None
        if bars_since_bull_sweep > fvgWaitBars:
            last_bull_sweep_bar = None
        
        # Entry conditions (only when no position - checking barstate.isconfirmed means bar is complete)
        in_session = True  # session filter always true in Python conversion
        
        # Short entry: Bear sweep + Bearish FVG
        short_setup = (bars_since_bear_sweep > 0 and bars_since_bear_sweep <= fvgWaitBars)
        short_entry = short_setup and bearish_fvg_valid and in_session
        
        # Long entry: Bull sweep + Bullish FVG
        long_setup = (bars_since_bull_sweep > 0 and bars_since_bull_sweep <= fvgWaitBars)
        long_entry = long_setup and bullish_fvg_valid and in_session
        
        # Execute entries
        if long_entry:
            entry_price = close.iloc[i]
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            
            trade_num += 1
            last_bull_sweep_bar = None
        
        if short_entry:
            entry_price = close.iloc[i]
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            
            trade_num += 1
            last_bear_sweep_bar = None

    return entries