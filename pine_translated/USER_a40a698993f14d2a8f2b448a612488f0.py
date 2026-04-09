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
    trade_num = 0
    
    close = df['close']
    open_price = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    timestamps = df['time']
    
    # Helper functions
    def is_up(idx):
        return close.iloc[idx] > open_price.iloc[idx]
    
    def is_down(idx):
        return close.iloc[idx] < open_price.iloc[idx]
    
    def is_ob_up(idx):
        return (is_down(idx + 1) and is_up(idx) and close.iloc[idx] > high.iloc[idx + 1])
    
    def is_ob_down(idx):
        return (is_up(idx + 1) and is_down(idx) and close.iloc[idx] < low.iloc[idx + 1])
    
    def is_fvg_up(idx):
        return (low.iloc[idx] > high.iloc[idx + 2])
    
    def is_fvg_down(idx):
        return (high.iloc[idx] < low.iloc[idx + 2])
    
    # Calculate time-based hour
    hours = pd.to_datetime(timestamps, unit='s', utc=True).hour
    
    # Input filters
    inp1 = False  # Volume Filter
    inp2 = False  # ATR Filter
    inp3 = False  # Trend Filter
    
    # Volume filter
    volfilt = True if not inp1 else (volume.shift(1) > volume.rolling(9).mean() * 1.5)
    
    # ATR filter (Wilder)
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    atr = tr.ewm(alpha=1/20, adjust=False).mean() / 1.5
    
    if inp2:
        atrfilt = (low - high.shift(2) > atr) | (low.shift(2) - high > atr)
    else:
        atrfilt = pd.Series([True] * len(df))
    
    # Trend filter
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    
    if inp3:
        locfiltb = loc2
        locfilts = ~loc2
    else:
        locfiltb = pd.Series([True] * len(df))
        locfilts = pd.Series([True] * len(df))
    
    # FVG conditions
    bfvg = (low > high.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (high < low.shift(2)) & volfilt & atrfilt & locfilts
    
    # OB and FVG indicators (shifted for lookback)
    obUp = pd.Series([False] * len(df))
    obDown = pd.Series([False] * len(df))
    fvgUp = pd.Series([False] * len(df))
    fvgDown = pd.Series([False] * len(df))
    
    for i in range(3, len(df)):
        obUp.iloc[i] = is_ob_up(1)
        obDown.iloc[i] = is_ob_down(1)
        fvgUp.iloc[i] = is_fvg_up(0)
        fvgDown.iloc[i] = is_fvg_down(0)
    
    # Time filter
    isValidTradeTime = (hours >= 10) & (hours < 12)
    
    # Imbalance conditions
    TopImbalance_Bway = (low.shift(2) <= open_price.shift(1)) & (high >= close.shift(1)) & (close < low.shift(1))
    Top_ImbXBway = (low.shift(2) <= open_price.shift(1)) & (high >= close.shift(1)) & (close > low.shift(1))
    BottomInbalance_Bway = (high.shift(2) >= open_price.shift(1)) & (low <= close.shift(1)) & (close > high.shift(1))
    Bottom_ImbXBAway = (high.shift(2) >= open_price.shift(1)) & (low <= close.shift(1)) & (close < high.shift(1))
    
    # Entry conditions
    long_condition = (obDown | fvgDown) & isValidTradeTime & (sfvg | BottomInbalance_Bway | Bottom_ImbXBAway)
    short_condition = (obUp | fvgUp) & isValidTradeTime & (bfvg | TopImbalance_Bway | Top_ImbXBway)
    
    # Generate entries
    for i in range(len(df)):
        if pd.isna(obUp.iloc[i]) or pd.isna(obDown.iloc[i]) or pd.isna(fvgUp.iloc[i]) or pd.isna(fvgDown.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            trade_num += 1
            entry_price = close.iloc[i]
            ts = int(timestamps.iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
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
        
        if short_condition.iloc[i]:
            trade_num += 1
            entry_price = close.iloc[i]
            ts = int(timestamps.iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
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
    
    return results