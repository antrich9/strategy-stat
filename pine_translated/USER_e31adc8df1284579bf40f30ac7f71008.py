def generate_entries(df: pd.DataFrame) -> list:
    # Ensure we have enough data
    if len(df) < 5:
        return []
    
    # Detect FVGs
    bullish_fvg_mask = df['low'] >= df['high'].shift(2)
    bearish_fvg_mask = df['low'].shift(2) >= df['high']
    
    # FVG boundaries
    bullish_fvg_top = df['low'].copy()
    bullish_fvg_bottom = df['high'].shift(2).copy()
    bearish_fvg_top = df['low'].shift(2).copy()
    bearish_fvg_bottom = df['high'].copy()
    
    entries = []
    trade_num = 1
    
    # Track active FVGs: list of tuples (idx, top, bottom, entered)
    active_bull_fvgs = []
    active_bear_fvgs = []
    
    # Iterate bars
    for i in range(2, len(df)):
        # Check for new FVG formation (on bar i, using data up to i)
        if bullish_fvg_mask.iloc[i]:
            active_bull_fvgs.append({
                'idx': i,
                'top': df['low'].iloc[i],
                'bottom': df['high'].iloc[i-2],
                'entered': False
            })
        
        if bearish_fvg_mask.iloc[i]:
            active_bear_fvgs.append({
                'idx': i,
                'top': df['low'].iloc[i-2],
                'bottom': df['high'].iloc[i],
                'entered': False
            })
        
        # Check for entry into active bullish FVGs
        current_low = df['low'].iloc[i]
        current_high = df['high'].iloc[i]
        
        for fvg in active_bull_fvgs:
            if not fvg['entered'] and fvg['bottom'] <= current_low <= fvg['top']:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_idx': i,
                    'entry_price': current_low,
                    'fvg_idx': fvg['idx']
                })
                fvg['entered'] = True
                trade_num += 1
        
        # Check for entry into active bearish FVGs
        for fvg in active_bear_fvgs:
            if not fvg['entered'] and fvg['top'] <= current_high <= fvg['bottom']:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_idx': i,
                    'entry_price': current_high,
                    'fvg_idx': fvg['idx']
                })
                fvg['entered'] = True
                trade_num += 1
        
        # Remove entered or mitigated FVGs from active list
        active_bull_fvgs = [fvg for fvg in active_bull_fvgs if not fvg['entered']]
        active_bear_fvgs = [fvg for fvg in active_bear_fvgs if not fvg['entered']]
    
    return entries