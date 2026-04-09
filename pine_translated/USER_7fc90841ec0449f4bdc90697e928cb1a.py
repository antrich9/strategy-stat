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
    results = []
    trade_num = 1

    # Helper functions
    def is_up(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]

    def is_down(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]

    def is_ob_up(idx):
        return (is_down(idx + 1) and is_up(idx) and 
                df['close'].iloc[idx] > df['high'].iloc[idx + 1])

    def is_ob_down(idx):
        return (is_up(idx + 1) and is_down(idx) and 
                df['close'].iloc[idx] < df['low'].iloc[idx + 1])

    def is_fvg_up(idx):
        return (df['low'].iloc[idx] > df['high'].iloc[idx + 2])

    def is_fvg_down(idx):
        return (df['high'].iloc[idx] < df['low'].iloc[idx + 2])

    # Volume filter
    vol_sma = df['volume'].rolling(9).mean()
    volfilt = vol_sma * 1.5 < df['volume'].shift(1)

    # ATR filter (Wilder)
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    atr = pd.Series(tr).rolling(20).mean().mul(1 / 1.5) if hasattr(pd.Series(tr), 'ewm') else tr.rolling(20).mean() * (1 / 1.5)
    atrfilt = (df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr)

    # Trend filter
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    # FVG conditions
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts

    # OB and FVG for signal
    obUp = is_ob_up(1)
    obDown = is_ob_down(1)
    fvgUp = is_fvg_up(0)
    fvgDown = is_fvg_down(0)

    # Time window detection (simplified - using hour extraction)
    times = pd.to_datetime(df['time'], unit='s', utc=True)
    hours = times.dt.hour + times.dt.minute / 60.0
    
    # Morning window: 2:45 to 5:45
    is_within_morning = (hours >= 2.75) & (hours < 5.75)
    # Afternoon window: 8:45 to 10:45
    is_within_afternoon = (hours >= 8.75) & (hours < 10.75)
    is_within_time_window = is_within_morning | is_within_afternoon

    # Combined entry conditions
    long_condition = bfvg & is_within_time_window
    short_condition = sfvg & is_within_time_window

    # Generate entries
    for i in range(len(df)):
        if i < 3:  # Skip first bars due to lookbacks
            continue
        
        if pd.isna(bfvg.iloc[i]) or pd.isna(sfvg.iloc[i]):
            continue
        
        entry_price = df['close'].iloc[i]
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        if long_condition.iloc[i]:
            results.append({
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
        
        if short_condition.iloc[i]:
            results.append({
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

    return results