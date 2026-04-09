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
    
    close = df['close']
    open_price = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Previous day's high and low (daily values shifted back by 1 day)
    prevDayHigh = high.shift(1)
    prevDayLow = low.shift(1)
    
    # Helper functions for Wilder RSI and ATR
    def wilder_rsi(src, length):
        delta = src.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def wilder_atr(length):
        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/length, adjust=False).mean()
        return atr
    
    # Volume filter
    volfilt = volume.shift(1) > volume.rolling(9).mean() * 1.5
    
    # ATR filter
    atr = wilder_atr(20) / 1.5
    atrfilt = ((low - high.shift(2) > atr) | (low.shift(2) - high > atr))
    
    # Trend filter
    loc = close.rolling(54).mean()
    locfiltb = loc > loc.shift(1)
    locfilts = loc <= loc.shift(1)
    
    # Bullish FVG (breakaway): low > high[2]
    bfvg = (low > high.shift(2)) & volfilt & atrfilt & locfiltb
    
    # Bearish FVG: high < low[2]
    sfvg = (high < low.shift(2)) & volfilt & atrfilt & locfilts
    
    # Imbalance conditions
    TopImbalance_Bway = (low.shift(2) <= open_price.shift(1)) & (high >= close.shift(1)) & (close < low.shift(1))
    BottomInbalance_Bway = (high.shift(2) >= open_price.shift(1)) & (low <= close.shift(1)) & (close > high.shift(1))
    
    # Cross conditions
    crossover_long = (close > prevDayLow) & (close.shift(1) <= prevDayLow.shift(1))
    crossunder_short = (close < prevDayHigh) & (close.shift(1) >= prevDayHigh.shift(1))
    
    # Entry conditions: liquidity sweep + FVG/OB pattern
    long_condition = crossover_long & (bfvg | TopImbalance_Bway)
    short_condition = crossunder_short & (sfvg | BottomInbalance_Bway)
    
    # Initialize output
    entries = []
    trade_num = 0
    
    # Iterate through bars (start from index 2 to ensure enough history)
    for i in range(2, len(df)):
        # Skip if required indicators are NaN
        if pd.isna(prevDayHigh.iloc[i]) or pd.isna(prevDayLow.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
        
        elif short_condition.iloc[i]:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
    
    return entries