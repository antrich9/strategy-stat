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
    n = len(df)
    if n < 5:
        return []
    
    entries = []
    trade_num = 0
    
    # Calculate Wilder ATR(144) for FVG filtering
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = np.zeros(n)
    atr[143] = tr.iloc[:144].mean()
    multiplier = 1.0 / 144
    for i in range(144, n):
        atr[i] = atr[i-1] * (1 - multiplier) + tr.iloc[i] * multiplier
    
    # Calculate Wilder ATR(14) for stop loss
    atr14 = np.zeros(n)
    atr14[13] = tr.iloc[:14].mean()
    multiplier14 = 1.0 / 14
    for i in range(14, n):
        atr14[i] = atr14[i-1] * (1 - multiplier14) + tr.iloc[i] * multiplier14
    
    # Previous day high/low (simplified - use daily resample or approximate)
    # Since we don't have daily data structure, we'll use a simpler approach
    # Get previous day's high/low from the data
    prev_day_high = np.zeros(n)
    prev_day_low = np.zeros(n)
    
    # Convert timestamps to datetime for day grouping
    times = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Calculate daily high/low
    daily_groups = times.dt.date
    daily_high = df['high'].groupby(daily_groups).transform('max')
    daily_low = df['low'].groupby(daily_groups).transform('min')
    
    prev_day_high[1:] = daily_high.shift(1).iloc[1:]
    prev_day_low[1:] = daily_low.shift(1).iloc[1:]
    
    # Check if previous day high or low has been swept
    previous_day_high_taken = df['high'] > prev_day_high
    previous_day_low_taken = df['low'] < prev_day_low
    
    # Flags for sweeping (var in Pine = persistent across bars)
    flagpdh = False
    flagpdl = False
    
    # Bull/Gap detection
    bullG = df['low'] > df['high'].shift(1)
    bearG = df['high'] < df['low'].shift(1)
    
    # FVG state variables
    fvg_upper = pd.Series(np.nan, index=df.index)
    fvg_lower = pd.Series(np.nan, index=df.index)
    fvg_exists = pd.Series(False, index=df.index)
    last_is_bull = False
    fvg_last_pct = 0.0
    
    # Midpoint tracking
    bull_midpoint = pd.Series(np.nan, index=df.index)
    bear_midpoint = pd.Series(np.nan, index=df.index)
    
    # Confirmed FVG lower/upper (updated on barstate.isconfirmed)
    confirmed_bull_fvg_lower = np.nan
    confirmed_bear_fvg_upper = np.nan
    
    # London trading windows
    times_dt = pd.to_datetime(df['time'], unit='s', utc=True)
    
    in_trading_window = pd.Series(False, index=df.index)
    for i in range(n):
        t = times_dt.iloc[i]
        hour = t.hour
        minute = t.minute
        total_minutes = hour * 60 + minute
        
        # Morning window: 07:45 - 09:45 UTC
        morning_start = 7 * 60 + 45
        morning_end = 9 * 60 + 45
        
        # Afternoon window: 14:45 - 16:45 UTC
        afternoon_start = 14 * 60 + 45
        afternoon_end = 16 * 60 + 45
        
        is_morning = morning_start <= total_minutes < morning_end
        is_afternoon = afternoon_start <= total_minutes < afternoon_end
        in_trading_window.iloc[i] = is_morning or is_afternoon
    
    # Detect FVG formations
    bull_fvg_detected = pd.Series(False, index=df.index)
    bear_fvg_detected = pd.Series(False, index=df.index)
    
    for i in range(2, n):
        if i < 144:
            continue
        
        atr_val = atr[i]
        if np.isnan(atr_val) or atr_val == 0:
            continue
        
        bullG_val = bullG.iloc[i]
        bullG_prev = bullG.iloc[i-1] if i > 0 else False
        
        bearG_val = bearG.iloc[i]
        bearG_prev = bearG.iloc[i-1] if i > 0 else False
        
        cond1 = (df['low'].iloc[i] - df['high'].iloc[i-2]) > atr_val
        cond2 = df['low'].iloc[i] > df['high'].iloc[i-2]
        cond3 = df['close'].iloc[i-1] > df['high'].iloc[i-2]
        cond4 = not (bullG_val or bullG_prev)
        
        if cond1 and cond2 and cond3 and cond4:
            bull_fvg_detected.iloc[i] = True
            bull_midpoint.iloc[i] = (df['high'].iloc[i-2] + df['low'].iloc[i]) / 2
            last_is_bull = True
            fvg_exists.iloc[i] = True
            confirmed_bull_fvg_lower = df['low'].iloc[i]
            fvg_lower.iloc[i] = df['low'].iloc[i]
            fvg_upper.iloc[i] = df['high'].iloc[i-2]
        
        cond1_b = (df['low'].iloc[i-2] - df['high'].iloc[i]) > atr_val
        cond2_b = df['high'].iloc[i] < df['low'].iloc[i-2]
        cond3_b = df['close'].iloc[i-1] < df['low'].iloc[i-2]
        cond4_b = not (bearG_val or bearG_prev)
        
        if cond1_b and cond2_b and cond3_b and cond4_b:
            bear_fvg_detected.iloc[i] = True
            bear_midpoint.iloc[i] = (df['high'].iloc[i] + df['low'].iloc[i-2]) / 2
            last_is_bull = False
            fvg_exists.iloc[i] = True
            confirmed_bear_fvg_upper = df['high'].iloc[i]
            fvg_upper.iloc[i] = df['high'].iloc[i]
            fvg_lower.iloc[i] = df['low'].iloc[i-2]
        
        # Update FVG state (similar to fvg.update)
        if fvg_exists.iloc[i]:
            # Calculate lastPct similar to the script
            if not np.isnan(fvg_upper.iloc[i]) and not np.isnan(fvg_lower.iloc[i]):
                fvg_last_pct = 0.5  # Simplified - using midpoint as proxy
    
    # Entry detection
    for i in range(1, n):
        # Update flags
        if previous_day_high_taken.iloc[i]:
            flagpdh = True
        if previous_day_low_taken.iloc[i]:
            flagpdl = True
        if previous_day_low_taken.iloc[i] or previous_day_high_taken.iloc[i]:
            pass
        else:
            flagpdl = False
            flagpdh = False
        
        # Check if FVG exists at this bar
        has_fvg = fvg_exists.iloc[i]
        
        if not has_fvg or np.isnan(fvg_last_pct) or fvg_last_pct <= 0.01 or fvg_last_pct > 1:
            continue
        
        if not in_trading_window.iloc[i]:
            continue
        
        in_trading = in_trading_window.iloc[i]
        
        # Determine if we're in a bull or bear FVG context
        is_bull_context = bull_fvg_detected.iloc[i] or (i > 0 and bull_fvg_detected.iloc[i-1])
        is_bear_context = bear_fvg_detected.iloc[i] or (i > 0 and bear_fvg_detected.iloc[i-1])
        
        bull_mid = bull_midpoint.iloc[i] if not np.isnan(bull_midpoint.iloc[i]) else np.nan
        bear_mid = bear_midpoint.iloc[i] if not np.isnan(bear_midpoint.iloc[i]) else np.nan
        
        # Get confirmed FVG boundaries
        bull_lower = fvg_lower.iloc[i] if is_bull_context else np.nan
        bear_upper = fvg_upper.iloc[i] if is_bear_context else np.nan
        
        # Check for crossunder (bullish entry trigger)
        if is_bull_context and not np.isnan(bull_lower):
            if i > 0:
                if df['low'].iloc[i] < bull_lower and df['low'].iloc[i-1] >= bull_lower:
                    # Bullish entry triggered (crossunder of bullFvgLower)
                    if flagpdl and not np.isnan(bull_mid) and df['low'].iloc[i] > bull_mid:
                        trade_num += 1
                        entry_ts = int(df['time'].iloc[i])
                        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                        entries.append({
                            'trade_num': trade_num,
                            'direction': 'long',
                            'entry_ts': entry_ts,
                            'entry_time': entry_time,
                            'entry_price_guess': df['close'].iloc[i],
                            'exit_ts': 0,
                            'exit_time': '',
                            'exit_price_guess': 0.0,
                            'raw_price_a': df['close'].iloc[i],
                            'raw_price_b': bull_lower
                        })
        
        # Check for crossover (bearish entry trigger)
        if is_bear_context and not np.isnan(bear_upper):
            if i > 0:
                if df['high'].iloc[i] > bear_upper and df['high'].iloc[i-1] <= bear_upper:
                    # Bearish entry triggered (crossover of bearFvgUpper)
                    if flagpdh and not np.isnan(bear_mid) and df['high'].iloc[i] < bear_mid:
                        trade_num += 1
                        entry_ts = int(df['time'].iloc[i])
                        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                        entries.append({
                            'trade_num': trade_num,
                            'direction': 'short',
                            'entry_ts': entry_ts,
                            'entry_time': entry_time,
                            'entry_price_guess': df['close'].iloc[i],
                            'exit_ts': 0,
                            'exit_time': '',
                            'exit_price_guess': 0.0,
                            'raw_price_a': df['close'].iloc[i],
                            'raw_price_b': bear_upper
                        })
    
    return entries