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
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Calculate ATR using Wilder's method
    def calc_wilder_atr(high, low, close, period):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    atr_144 = calc_wilder_atr(df['high'], df['low'], df['close'], 144)
    atr_14 = calc_wilder_atr(df['high'], df['low'], df['close'], 14)
    
    # Previous day high and low using daily resample
    df['_dt'] = pd.to_datetime(df['time'], unit='s')
    df['_day'] = df['_dt'].dt.date
    daily_data = df.groupby('_day').agg({'high': 'max', 'low': 'min'}).reset_index()
    daily_data.columns = ['_day', '_day_high', '_day_low']
    df = df.merge(daily_data, on='_day')
    df['_prev_day_high'] = df['_day_high'].shift(1)
    df['_prev_day_low'] = df['_day_low'].shift(1)
    
    # Previous day high/low swept
    prev_day_high_taken = df['high'] > df['_prev_day_high']
    prev_day_low_taken = df['low'] < df['_prev_day_low']
    
    # Track flags (var bool behavior)
    flagpdh = pd.Series(False, index=df.index)
    flagpdl = pd.Series(False, index=df.index)
    
    for i in range(1, len(df)):
        if prev_day_high_taken.iloc[i]:
            flagpdh.iloc[i] = True
            flagpdl.iloc[i] = False
        elif prev_day_low_taken.iloc[i]:
            flagpdl.iloc[i] = True
            flagpdh.iloc[i] = False
        else:
            flagpdh.iloc[i] = flagpdh.iloc[i-1]
            flagpdl.iloc[i] = flagpdl.iloc[i-1]
    
    # FVG settings
    fvgTH = 0.5
    atr_threshold = atr_144 * fvgTH
    
    # Bullish/Bearish gap detection
    bullG = df['low'] > df['high'].shift(1)
    bearG = df['high'] < df['low'].shift(1)
    
    # Bullish FVG condition
    bull_fvg_cond = (
        (df['low'] - df['high'].shift(2)) > atr_threshold
    ) & (
        df['low'] > df['high'].shift(2)
    ) & (
        df['close'].shift(1) > df['high'].shift(2)
    ) & ~(
        bullG | bullG.shift(1)
    )
    
    # Bearish FVG condition
    bear_fvg_cond = (
        (df['low'].shift(2) - df['high']) > atr_threshold
    ) & (
        df['high'] < df['low'].shift(2)
    ) & (
        df['close'].shift(1) < df['low'].shift(2)
    ) & ~(
        bearG | bearG.shift(1)
    )
    
    # FVG upper/lower bounds
    bullFvgUpper = pd.Series(np.nan, index=df.index)
    bullFvgLower = pd.Series(np.nan, index=df.index)
    bearFvgUpper = pd.Series(np.nan, index=df.index)
    bearFvgLower = pd.Series(np.nan, index=df.index)
    
    bullFvgUpper[bull_fvg_cond] = df['high'].shift(2)[bull_fvg_cond]
    bullFvgLower[bull_fvg_cond] = df['low'][bull_fvg_cond]
    bearFvgUpper[bear_fvg_cond] = df['high'][bear_fvg_cond]
    bearFvgLower[bear_fvg_cond] = df['low'].shift(2)[bear_fvg_cond]
    
    # Midpoints
    bullMidpoint = pd.Series(np.nan, index=df.index)
    bearMidpoint = pd.Series(np.nan, index=df.index)
    
    # FVG last percentage (simplified tracking)
    fvg_lastPct = pd.Series(np.nan, index=df.index)
    
    # Track FVG state
    fvg_exists = pd.Series(False, index=df.index)
    last_dir = pd.Series(dtype=float)
    
    for i in range(2, len(df)):
        # Update FVG percentage when updating
        if fvg_exists.iloc[i-1]:
            ub = bullFvgUpper.iloc[i-1] if not pd.isna(bullFvgUpper.iloc[i-1]) else bearFvgUpper.iloc[i-1]
            lb = bullFvgLower.iloc[i-1] if not pd.isna(bullFvgLower.iloc[i-1]) else bearFvgLower.iloc[i-1]
            mid_b = bullFvgLower.iloc[i-1] if not pd.isna(bullFvgLower.iloc[i-1]) else bearFvgUpper.iloc[i-1]
            if not pd.isna(ub) and not pd.isna(lb) and (ub - lb) != 0:
                fvg_lastPct.iloc[i] = (mid_b - lb) / (ub - lb) if last_dir.iloc[i-1] == 1 else (ub - mid_b) / (ub - lb)
        
        # Clear FVG on gap
        if bullG.iloc[i] or bearG.iloc[i]:
            fvg_exists.iloc[i] = False
            bullFvgUpper.iloc[i] = np.nan
            bullFvgLower.iloc[i] = np.nan
            bearFvgUpper.iloc[i] = np.nan
            bearFvgLower.iloc[i] = np.nan
        else:
            fvg_exists.iloc[i] = fvg_exists.iloc[i-1]
            bullFvgUpper.iloc[i] = bullFvgUpper.iloc[i-1]
            bullFvgLower.iloc[i] = bullFvgLower.iloc[i-1]
            bearFvgUpper.iloc[i] = bearFvgUpper.iloc[i-1]
            bearFvgLower.iloc[i] = bearFvgLower.iloc[i-1]
        
        # Set new FVG
        if bull_fvg_cond.iloc[i]:
            fvg_exists.iloc[i] = True
            bullFvgUpper.iloc[i] = df['high'].iloc[i-2]
            bullFvgLower.iloc[i] = df['low'].iloc[i]
            bearFvgUpper.iloc[i] = np.nan
            bearFvgLower.iloc[i] = np.nan
            last_dir.iloc[i] = 1.0
        elif bear_fvg_cond.iloc[i]:
            fvg_exists.iloc[i] = True
            bearFvgUpper.iloc[i] = df['high'].iloc[i]
            bearFvgLower.iloc[i] = df['low'].iloc[i-2]
            bullFvgUpper.iloc[i] = np.nan
            bullFvgLower.iloc[i] = np.nan
            last_dir.iloc[i] = 0.0
        else:
            last_dir.iloc[i] = last_dir.iloc[i-1] if not pd.isna(last_dir.iloc[i-1]) else np.nan
        
        # Calculate midpoint on confirmed bar
        if fvg_exists.iloc[i]:
            if last_dir.iloc[i] == 1.0 and not pd.isna(bullFvgUpper.iloc[i]) and not pd.isna(bullFvgLower.iloc[i]):
                bullMidpoint.iloc[i] = (bullFvgUpper.iloc[i] + bullFvgLower.iloc[i]) / 2
            elif last_dir.iloc[i] == 0.0 and not pd.isna(bearFvgUpper.iloc[i]) and not pd.isna(bearFvgLower.iloc[i]):
                bearMidpoint.iloc[i] = (bearFvgUpper.iloc[i] + bearFvgLower.iloc[i]) / 2
    
    # Forward fill last direction
    last_dir = last_dir.ffill()
    
    # Time window checks (London time)
    df['_hour'] = df['_dt'].dt.hour
    df['_minute'] = df['_dt'].dt.minute
    
    is_morning = (
        ((df['_hour'] == 7) & (df['_minute'] >= 45)) |
        ((df['_hour'] == 8)) |
        ((df['_hour'] == 9) & (df['_minute'] < 45))
    )
    
    is_afternoon = (
        ((df['_hour'] == 14) & (df['_minute'] >= 45)) |
        ((df['_hour'] == 15)) |
        ((df['_hour'] == 16) & (df['_minute'] < 45))
    )
    
    in_trading_window = is_morning | is_afternoon
    
    # Build entry conditions
    bull_entry_cond = (
        fvg_exists & 
        (fvg_lastPct > 0.01) & 
        (fvg_lastPct <= 1) & 
        (last_dir == 1.0) & 
        in_trading_window & 
        flagpdl & 
        bullMidpoint.notna() & 
        (df['low'] > bullMidpoint)
    )
    
    bear_entry_cond = (
        fvg_exists & 
        (fvg_lastPct > 0.01) & 
        (fvg_lastPct <= 1) & 
        (last_dir == 0.0) & 
        in_trading_window & 
        flagpdh & 
        bearMidpoint.notna() & 
        (df['high'] < bearMidpoint)
    )
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if bull_entry_cond.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif bear_entry_cond.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries