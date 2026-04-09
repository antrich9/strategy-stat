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
    
    # Input parameters (matching Pine Script defaults)
    lookback_bars = 12
    threshold = 0.0
    
    # Volume filter: volume > sma(volume, 9) * 1.5
    volfilt = df['volume'] > df['volume'].rolling(9).mean() * 1.5
    
    # ATR filter: (low - high[2] > atr) or (low[2] - high > atr) where atr = ta.atr(20) / 1.5
    atr = compute_atr(df, 20) / 1.5
    atrfilt = (df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr)
    
    # Trend filter using 54 SMA
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # Bullish FVG: low > high[2] and close[1] > high[2]
    bull_fvg = (df['low'] > df['high'].shift(2)) & (df['close'].shift(1) > df['high'].shift(2))
    
    # Bearish FVG: high < low[2] and close[1] < low[2]
    bear_fvg = (df['high'] < df['low'].shift(2)) & (df['close'].shift(1) < df['low'].shift(2))
    
    # Bullish FVG with all filters
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    
    # Bearish FVG with all filters
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    # Implement barssince for bull_since (bars since bull_fvg)
    bull_since = barssince_series(bull_fvg)
    
    # Bull cond 1: bull_fvg and bull_since <= lookback_bars
    bull_cond_1 = bull_fvg & (bull_since <= lookback_bars)
    
    # Combined levels for BPR calculation
    combined_low_bull = np.where(bull_cond_1, df['high'].shift(1), np.nan)
    combined_high_bull = np.where(bull_cond_1, df['low'], np.nan)
    
    # Bull result: bull_cond_1 and (combined_high_bull - combined_low_bull >= threshold)
    bull_result = bull_cond_1 & ((combined_high_bull - combined_low_bull) >= threshold)
    
    # Implement barssince for bear_since (bars since bear_fvg)
    bear_since = barssince_series(bear_fvg)
    
    # Bear cond 1: bear_fvg and bear_since <= lookback_bars
    bear_cond_1 = bear_fvg & (bear_since <= lookback_bars)
    
    # Combined levels for BPR calculation
    combined_low_bear = np.where(bear_cond_1, df['high'], np.nan)
    combined_high_bear = np.where(bear_cond_1, df['low'].shift(1), np.nan)
    
    # Bear result: bear_cond_1 and (combined_high_bear - combined_low_bear >= threshold)
    bear_result = bear_cond_1 & ((combined_high_bear - combined_low_bear) >= threshold)
    
    # Long entry conditions
    long_cond = bull_fvg | bull_result
    
    # Short entry conditions
    short_cond = bear_fvg | bear_result
    
    # Build entry signals
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(long_cond.iloc[i]) or pd.isna(short_cond.iloc[i]):
            continue
        
        if long_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
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
            trade_num += 1
        elif short_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
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
            trade_num += 1
    
    return entries

def barssince_series(condition: pd.Series) -> pd.Series:
    """
    Calculate barssince equivalent: number of bars since condition was true.
    Returns NaN where condition has never been true yet.
    """
    result = pd.Series(index=condition.index, dtype=float)
    count = np.nan
    for i in range(len(condition)):
        if condition.iloc[i]:
            count = 0
        elif not pd.isna(count):
            count += 1
        result.iloc[i] = count
    return result

def compute_atr(df: pd.DataFrame, length: int) -> pd.Series:
    """
    Compute Average True Range using Wilder's smoothing method.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    
    return atr