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
    # ATR calculation (Wilder)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    
    atr_144 = np.zeros(len(df))
    atr_144[0] = np.nan
    alpha = 1.0 / 144.0
    for i in range(1, len(df)):
        if i == 1:
            atr_144[i] = tr[1]
        else:
            if np.isnan(atr_144[i-1]):
                atr_144[i] = np.nan
            else:
                atr_144[i] = alpha * tr[i] + (1 - alpha) * atr_144[i-1]
    
    fvgTH = 0.5
    atr_threshold = atr_144 * fvgTH
    
    # Bullish FVG condition: (b.l - b.h[2]) > atr and b.l > b.h[2] and b.c[1] > b.h[2] and not (bullG or bullG[1])
    bullG = low > np.roll(high, 1)
    bullG[0] = False
    
    bullG_1 = np.roll(bullG, 1)
    bullG_1[0] = False
    
    bull_fvg_cond = (low - np.roll(high, 2)) > atr_threshold
    bull_fvg_cond = bull_fvg_cond & (low > np.roll(high, 2))
    bull_fvg_cond = bull_fvg_cond & (np.roll(close, 1) > np.roll(high, 2))
    bull_fvg_cond = bull_fvg_cond & ~(bullG | bullG_1)
    
    # Bearish FVG condition: (b.l[2] - b.h) > atr and b.h < b.l[2] and b.c[1] < b.l[2] and not (bearG or bearG[1])
    bearG = high < np.roll(low, 1)
    bearG[0] = False
    
    bearG_1 = np.roll(bearG, 1)
    bearG_1[0] = False
    
    bear_fvg_cond = (np.roll(low, 2) - high) > atr_threshold
    bear_fvg_cond = bear_fvg_cond & (high < np.roll(low, 2))
    bear_fvg_cond = bear_fvg_cond & (np.roll(close, 1) < np.roll(low, 2))
    bear_fvg_cond = bear_fvg_cond & ~(bearG | bearG_1)
    
    # Calculate midpoints
    bullFvgUpper = np.where(bull_fvg_cond, np.roll(high, 2), np.nan)
    bullFvgLower = np.where(bull_fvg_cond, low, np.nan)
    bullMidpoint = (bullFvgUpper + bullFvgLower) / 2
    
    bearFvgUpper = np.where(bear_fvg_cond, high, np.nan)
    bearFvgLower = np.where(bear_fvg_cond, np.roll(low, 2), np.nan)
    bearMidpoint = (bearFvgUpper + bearFvgLower) / 2
    
    # Track FVG state
    bullFvgUpper_arr = np.zeros(len(df))
    bullFvgUpper_arr[:] = np.nan
    bullFvgLower_arr = np.zeros(len(df))
    bullFvgLower_arr[:] = np.nan
    bullMidpoint_arr = np.zeros(len(df))
    bullMidpoint_arr[:] = np.nan
    
    bearFvgUpper_arr = np.zeros(len(df))
    bearFvgUpper_arr[:] = np.nan
    bearFvgLower_arr = np.zeros(len(df))
    bearFvgLower_arr[:] = np.nan
    bearMidpoint_arr = np.zeros(len(df))
    bearMidpoint_arr[:] = np.nan
    
    last_fvg_type = np.zeros(len(df))
    last_fvg_type[:] = np.nan
    fvg_lastPct = np.zeros(len(df))
    fvg_lastPct[:] = np.nan
    fvg_size = np.zeros(len(df))
    fvg_size[:] = np.nan
    
    # Simplified FVG tracking
    for i in range(len(df)):
        if bull_fvg_cond[i]:
            bullFvgUpper_arr[i] = np.roll(high, 2)[i]
            bullFvgLower_arr[i] = low[i]
            bullMidpoint_arr[i] = (bullFvgUpper_arr[i] + bullFvgLower_arr[i]) / 2
            fvg_size[i] = 1
            last_fvg_type[i] = 1  # Bullish
        elif bear_fvg_cond[i]:
            bearFvgUpper_arr[i] = high[i]
            bearFvgLower_arr[i] = np.roll(low, 2)[i]
            bearMidpoint_arr[i] = (bearFvgUpper_arr[i] + bearFvgLower_arr[i]) / 2
            fvg_size[i] = 1
            last_fvg_type[i] = 0  # Bearish
        else:
            if i > 0:
                fvg_size[i] = fvg_size[i-1]
                last_fvg_type[i] = last_fvg_type[i-1]
                bullFvgUpper_arr[i] = bullFvgUpper_arr[i-1]
                bullFvgLower_arr[i] = bullFvgLower_arr[i-1]
                bullMidpoint_arr[i] = bullMidpoint_arr[i-1]
                bearFvgUpper_arr[i] = bearFvgUpper_arr[i-1]
                bearFvgLower_arr[i] = bearFvgLower_arr[i-1]
                bearMidpoint_arr[i] = bearMidpoint_arr[i-1]
    
    # Calculate fvg.lastPct based on mitigation
    for i in range(len(df)):
        if fvg_size[i] > 0 and not np.isnan(last_fvg_type[i]):
            if last_fvg_type[i] == 1:  # Bullish FVG active
                upper = bullFvgUpper_arr[i]
                lower = bullFvgLower_arr[i]
                if not np.isnan(upper) and not np.isnan(lower) and (upper - lower) > 0:
                    fvg_lastPct[i] = min(1.0, max(0.0, (close[i] - lower) / (upper - lower)))
            elif last_fvg_type[i] == 0:  # Bearish FVG active
                upper = bearFvgUpper_arr[i]
                lower = bearFvgLower_arr[i]
                if not np.isnan(upper) and not np.isnan(lower) and (upper - lower) > 0:
                    fvg_lastPct[i] = min(1.0, max(0.0, (upper - close[i]) / (upper - lower)))
    
    # Previous day high/low logic
    times = df['time'].values
    prev_day_high = np.zeros(len(df))
    prev_day_low = np.zeros(len(df))
    prev_day_high[:] = np.nan
    prev_day_low[:] = np.nan
    
    # Group by day
    dates = np.array([datetime.utcfromtimestamp(t).date() for t in times])
    unique_dates = np.unique(dates)
    
    for i, d in enumerate(unique_dates[:-1]):
        day_mask = dates == d
        next_day_mask = dates == unique_dates[i + 1]
        if np.any(day_mask):
            prev_day_high[next_day_mask] = high[day_mask][-1]
            prev_day_low[next_day_mask] = low[day_mask][-1]
    
    # Flagpdl: previous day low has been swept
    flagpdl = low < prev_day_low
    flagpdl = flagpdl & ~np.isnan(prev_day_low)
    
    # London trading window
    in_trading_window = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        dt = datetime.utcfromtimestamp(times[i])
        hour = dt.hour
        minute = dt.minute
        total_minutes = hour * 60 + minute
        morning_start = 7 * 60 + 45  # 07:45
        morning_end = 9 * 60 + 45    # 09:45
        afternoon_start = 14 * 60 + 45  # 14:45
        afternoon_end = 16 * 60 + 45    # 16:45
        
        in_morning = total_minutes >= morning_start and total_minutes < morning_end
        in_afternoon = total_minutes >= afternoon_start and total_minutes < afternoon_end
        in_trading_window[i] = in_morning or in_afternoon
    
    # Crossunder(low, bullMidpoint) for bullish
    crossunder_bull = (low < bullMidpoint) & (np.roll(low, 1) >= np.roll(bullMidpoint, 1))
    crossunder_bull[0] = False
    
    # Crossunder(high, bearMidpoint) for bearish (if needed)
    crossunder_bear = (high > bearMidpoint) & (np.roll(high, 1) <= np.roll(bearMidpoint, 1))
    crossunder_bear[0] = False
    
    # Entry conditions
    entries = []
    trade_num = 1
    in_position = False
    
    for i in range(len(df)):
        ts = int(times[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        # Entry condition: no position, FVG size > 0, lastPct > 0.01 and <= 1
        if fvg_size[i] > 0 and fvg_lastPct[i] > 0.01 and fvg_lastPct[i] <= 1:
            # Bullish entry
            if last_fvg_type[i] == 1 and crossunder_bull[i] and in_trading_window[i] and flagpdl[i]:
                entry_price = bullMidpoint_arr[i]
                if not np.isnan(entry_price) and entry_price > 0:
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': ts,
                        'entry_time': entry_time,
                        'entry_price_guess': float(entry_price),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(entry_price),
                        'raw_price_b': float(entry_price)
                    })
                    trade_num += 1
                    in_position = True
            # Bearish entry (if needed)
            elif last_fvg_type[i] == 0 and crossunder_bear[i] and in_trading_window[i]:
                entry_price = bearMidpoint_arr[i]
                if not np.isnan(entry_price) and entry_price > 0:
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': ts,
                        'entry_time': entry_time,
                        'entry_price_guess': float(entry_price),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(entry_price),
                        'raw_price_b': float(entry_price)
                    })
                    trade_num += 1
                    in_position = True
        
        # Reset position flag when FVG cleared (bullG or bearG)
        if i > 0:
            if bullG[i] or bearG[i]:
                in_position = False
    
    return entries