def generate_entries(df: pd.DataFrame) -> list:
    # Ensure sorted
    df = df.sort_values('time').reset_index(drop=True)
    # Initialize
    entries = []
    trade_num = 1
    # Compute FVG detection
    low = df['low']
    high = df['high']
    close = df['close']
    time = df['time']
    # We need low[2] and high[2], so start from index 2
    x = pd.Series(0, index=df.index)
    # For i from 2 to len-1
    for i in range(2, len(df)):
        low_2 = low.iloc[i-2]
        high_2 = high.iloc[i-2]
        low_i = low.iloc[i]
        high_i = high.iloc[i]
        if low_2 >= high_i:
            x.iloc[i] = -1
        elif low_i >= high_2:
            x.iloc[i] = 1
        else:
            x.iloc[i] = 0
    # Generate entries
    for i in range(len(df)):
        if x.iloc[i] == 0:
            continue
        direction = 'long' if x.iloc[i] == 1 else 'short'
        entry_price = close.iloc[i]
        ts = time.iloc[i]
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entries.append({
            'trade_num': trade_num,
            'direction': direction,
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
    return entries