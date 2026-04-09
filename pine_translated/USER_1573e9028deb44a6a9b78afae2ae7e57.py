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
    
    # Initialize FVG tracking
    bull_tops = []
    bull_bottoms = []
    bull_ts = []
    bull_entered = []
    
    bear_tops = []
    bear_bottoms = []
    bear_ts = []
    bear_entered = []
    
    for i in range(2, len(df)):
        ts = df['time'].iloc[i]
        
        # London time check (placeholder - would need actual conversion)
        in_session = True  # Simplified for now
        
        if not in_session:
            continue
        
        current_low = df['low'].iloc[i]
        current_high = df['high'].iloc[i]
        prev_high = df['high'].iloc[i-1]
        prev_low = df['low'].iloc[i-1]
        high_2 = df['high'].iloc[i-2]
        low_2 = df['low'].iloc[i-2]
        
        # Detect new FVG formations
        if current_low >= high_2:
            bull_tops.append(current_low)
            bull_bottoms.append(high_2)
            bull_ts.append(ts)
            bull_entered.append(False)
        
        if current_high <= low_2:
            bear_tops.append(low_2)
            bear_bottoms.append(current_high)
            bear_ts.append(ts)
            bear_entered.append(False)
        
        # Process bullish FVG entries
        j = len(bull_tops) - 1
        while j >= 0:
            if not bull_entered[j]:
                if current_low < bull_bottoms[j]:
                    bull_entered[j] = True
                elif current_low < bull_tops[j]:
                    bull_entered[j] = True
            j -= 1
        
        # Process bearish FVG entries
        j = len(bear_tops) - 1
        while j >= 0:
            if not bear_entered[j]:
                if current_high > bear_tops[j]:
                    bear_entered[j] = True
                elif current_high > bear_bottoms[j]:
                    bear_entered[j] = True
            j -= 1