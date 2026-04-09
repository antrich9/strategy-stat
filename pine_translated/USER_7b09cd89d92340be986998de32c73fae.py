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
    
    # Time window definitions (London timezone)
    # Window 1: 07:45-11:45
    # Window 2: 13:45-14:45
    
    # Convert timestamps to datetime for window filtering
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('Europe/London')
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    
    # Window 1: 7*60+45=465 to 11*60+45=705
    window1 = (df['time_minutes'] >= 465) & (df['time_minutes'] < 705)
    # Window 2: 13*60+45=825 to 14*60+45=885
    window2 = (df['time_minutes'] >= 825) & (df['time_minutes'] < 885)
    
    in_trading_window = window1 | window2
    
    # Wilder RSI implementation
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Wilder ATR implementation
    def wilder_atr(high, low, close, length):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        return atr
    
    # ATR(200)
    atr200 = wilder_atr(df['high'], df['low'], df['close'], 200)
    
    # FVG Detection: Bullish FVG = gap between bar i-2 and bar i (middle bar is smaller)
    # For bullish: low of bar i > high of bar i-2 (imbalance going up)
    # For bearish: high of bar i < low of bar i-2 (imbalance going down)
    
    high_minus2 = df['high'].shift(2)
    low_minus2 = df['low'].shift(2)
    high_minus1 = df['high'].shift(1)
    low_minus1 = df['low'].shift(1)
    
    # Bullish FVG: middle bar high < previous bar low (price moved up through imbalance)
    # We check if the imbalance is still present (not filled)
    bullish_fvg_condition = (df['low'] > high_minus1)
    bearish_fvg_condition = (df['high'] < low_minus1)
    
    # Detect if bullish FVG is filled (price comes back down to fill the gap)
    bull_fvg_filled = (df['low'] <= high_minus1)
    bear_fvg_filled = (df['high'] >= low_minus1)
    
    # Detect FVG zones
    bullish_fvg = bullish_fvg_condition & in_trading_window
    bearish_fvg = bearish_fvg_condition & in_trading_window
    
    # Filled conditions
    bullish_fvg_filled_cond = bull_fvg_filled & in_trading_window
    bearish_fvg_filled_cond = bear_fvg_filled & in_trading_window
    
    # Tracking arrays for FVG fills (simplified - tracking last N)
    max_fvg_tracking = 200
    
    bull_fvg_count = 0
    bear_fvg_count = 0
    
    bull_fvg_levels = []
    bear_fvg_levels = []
    
    # Check for NaN in ATR (skip first bars until ATR is valid)
    valid_mask = ~atr200.isna()
    
    for i in range(2, len(df)):
        if not valid_mask.iloc[i]:
            continue
        if not in_trading_window.iloc[i]:
            continue
        
        # Check for new bullish FVG
        if bullish_fvg.iloc[i]:
            fvg_top = df['high'].iloc[i-1]
            fvg_bottom = df['low'].iloc[i-1]
            bull_fvg_levels.append(fvg_top)
            if len(bull_fvg_levels) > max_fvg_tracking:
                bull_fvg_levels.pop(0)
        
        # Check for new bearish FVG
        if bearish_fvg.iloc[i]:
            fvg_top = df['high'].iloc[i-1]
            fvg_bottom = df['low'].iloc[i-1]
            bear_fvg_levels.append(fvg_bottom)
            if len(bear_fvg_levels) > max_fvg_tracking:
                bear_fvg_levels.pop(0)
        
        # Check if any bullish FVG was filled
        if df['low'].iloc[i] <= df['high'].iloc[i-1] and bull_fvg_levels:
            bull_fvg_count += 1
            bull_fvg_levels.pop(0)
        
        # Check if any bearish FVG was filled
        if df['high'].iloc[i] >= df['low'].iloc[i-1] and bear_fvg_levels:
            bear_fvg_count += 1
            bear_fvg_levels.pop(0)
    
    return results