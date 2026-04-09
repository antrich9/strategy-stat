def generate_entries(df: pd.DataFrame) -> list:
    import pandas as pd
    import numpy as np
    from datetime import datetime, timezone

    # Ensure sorted
    df = df.sort_values('time').reset_index(drop=True)

    # Compute bar period in seconds
    if len(df) < 2:
        return []
    bar_period = int(np.median(np.diff(df['time'].values)))
    if bar_period == 0:
        bar_period = 1

    # HTF period in seconds (weekly)
    htf_period = 7 * 24 * 60 * 60  # weekly

    # Compute shift
    shift = int(htf_period / bar_period) + 1

    # Compute shifted high and low
    df['high_shift'] = df['high'].shift(shift)
    df['low_shift'] = df['low'].shift(shift)

    # Compute FVG conditions
    df['bullish_fvg'] = df['low'] > df['high_shift']
    df['bearish_fvg'] = df['high'] < df['low_shift']

    # Optionally filter by threshold (range)
    # threshold = 0, ignore

    # Generate entries
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if df['bullish_fvg'].iloc[i]:
            entry_ts = df['time'].iloc[i]
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
        elif df['bearish_fvg'].iloc[i]:
            entry_ts = df['time'].iloc[i]
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