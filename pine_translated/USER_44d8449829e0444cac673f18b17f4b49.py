import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.
    """
    
    # Calculate daily bias
    # Get previous day's high, low, close
    # For this, we need to identify daily boundaries
    
    # For simplicity, let's assume df might be intraday or daily
    # We'll calculate bias based on the data we have
    
    # Actually, looking at the Pine Script, it uses request.security for daily data
    # But we don't have that luxury in pure Python with just OHLCV
    # So we'll need to infer daily bias from the available data
    
    # Let's calculate bias using rolling windows of highs/lows/closes
    
    # Calculate bias
    # Bearish: prev_daily_close < prev_prev_daily_low
    # Bullish: prev_daily_close > prev_prev_daily_high
    # Ranging: in between
    
    # We need at least 2 periods to compare
    # Let's assume each row represents a period and calculate bias accordingly
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Shift to get previous values
    prev_daily_close = close.shift(1)
    prev_daily_high = high.shift(1)
    prev_daily_low = low.shift(1)
    
    prev_prev_daily_close = close.shift(2)
    prev_prev_daily_high = high.shift(2)
    prev_prev_daily_low = low.shift(2)
    
    # Calculate bias
    is_bearish = prev_daily_close < prev_prev_daily_low
    is_bullish = prev_daily_close > prev_prev_daily_high
    is_ranging = ~is_bearish & ~is_bullish
    
    bias = pd.Series('ranging', index=close.index)
    bias[is_bearish] = 'bearish'
    bias[is_bullish] = 'bullish'
    
    # FVG Detection
    # Bullish FVG: high[2] < low (gap up)
    # Bearish FVG: low[2] > high (gap down)
    
    bullish_fvg = high.shift(2) < low
    bearish_fvg = low.shift(2) > high
    
    # Direction
    # 1 = bullish FVG detected
    # -1 = bearish FVG detected
    ig_direction = pd.Series(0, index=close.index)
    ig_direction[bullish_fvg] = 1
    ig_direction[bearish_fvg] = -1
    
    # Store FVG high/low
    ig_c1_high = high.shift(2).where(bullish_fvg, low.shift(2).where(bearish_fvg, np.nan))
    ig_c1_low = low.shift(2).where(bearish_fvg, high.shift(2).where(bullish_fvg, np.nan))
    
    # Validation
    # Bullish FVG validated: close < ig_c1_high
    # Bearish FVG validated: close > ig_c1_low
    validated_bullish = (ig_direction == 1) & (close < ig_c1_high)
    validated_bearish = (ig_direction == -1) & (close > ig_c1_low)
    
    validated = validated_bullish | validated_bearish
    
    # System entry conditions
    system_entry_long = validated & (ig_direction == -1) & (bias == 'bearish')
    system_entry_short = validated & (ig_direction == 1) & (bias == 'bullish')
    
    # Trade direction
    trade_direction = pd.Series(index=close.index, dtype=object)
    trade_direction[system_entry_long] = 'long'
    trade_direction[system_entry_short] = 'short'
    
    return trade_direction