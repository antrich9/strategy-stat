def generate_entries(df: pd.DataFrame) -> list:
    # Calculate FVG conditions
    # bull_fvg = low > high[2] and close[1] > high[2]
    # bear_fvg = high < low[2] and close[1] < low[2]
    
    # Detect FVGs
    bull_fvg = (df['low'] > df['high'].shift(2)) & (df['close'].shift(1) > df['high'].shift(2))
    bear_fvg = (df['high'] < df['low'].shift(2)) & (df['close'].shift(1) < df['low'].shift(2))
    
    # For each FVG, we need to track it and check for entry
    # Entry for bull fvg: price (low) < FVG top (which is low at the time of FVG)
    # Entry for bear fvg: price (high) > FVG bottom (which is high at the time of FVG)