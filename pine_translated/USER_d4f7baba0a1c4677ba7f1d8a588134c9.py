def generate_entries(df: pd.DataFrame) -> list:
    df['day'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.date
    
    # Previous Day High/Low
    df['is_new_day'] = df['day'] != df['day'].shift(1)
    df['pdHigh'] = df['high'].where(df['is_new_day']).groupby(df['day']).transform('max').shift(1)
    df['pdLow'] = df['low'].where(df['is_new_day']).groupby(df['day']).transform('min').shift(1)
    
    # Sweep Detection
    df['sweepHigh'] = ~df['pdHigh'].isna() & (df['high'] > df['pdHigh'])
    df['sweepLow'] = ~df['pdLow'].isna() & (df['low'] < df['pdLow'])
    
    # Supertrend (simplified - using ATR-based approach)
    df['atr'] = ta_atr(df['high'], df['low'], df['close'], 14)
    df['st_upper'] = df['high'] + 3 * df['atr']
    df['st_lower'] = df['low'] - 3 * df['atr']
    df['st_direction'] = 1  # Initialize direction