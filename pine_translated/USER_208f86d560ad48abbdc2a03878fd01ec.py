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
    import pandas as pd
    import numpy as np
    from datetime import datetime, timezone

    # Make a copy to avoid mutating the original DataFrame
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()

    # Shifted series needed for FVG detection
    high_2 = df['high'].shift(2)
    open_1 = df['open'].shift(1)
    close_1 = df['close'].shift(1)
    high_1 = df['high'].shift(1)
    low_2 = df['low'].shift(2)

    # Bullish Breakaway FVG
    bullish_fvg = (high_2 >= open_1) & (df['low'] <= close_1) & (df['close'] > high_1)
    bullish_gap = df['low'] - high_2
    long_cond = bullish_fvg & (bullish_gap > 0)

    # Bearish Breakaway FVG
    bearish_fvg = (low_2 <= open_1) & (df['high'] >= close_1) & (df['close'] < low_2)
    bearish_gap = low_2 - df['high']
    short_cond = bearish_fvg & (bearish_gap > 0)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        # Skip bars where any required indicator is NaN (first two bars)
        if (pd.isna(high_2.iloc[i]) or pd.isna(open_1.iloc[i]) or
            pd.isna(close_1.iloc[i]) or pd.isna(high_1.iloc[i]) or
            pd.isna(low_2.iloc[i])):
            continue

        if long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1

        if short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1

    return entries