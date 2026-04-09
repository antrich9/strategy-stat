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

    # Calculate indicators
    ma1 = df['close'].rolling(200).mean()
    ma2 = df['close'].rolling(10).mean()

    # Date filter
    start_ts = int(datetime(1995, 1, 1, 13, 30, tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime(2099, 1, 1, 19, 30, tzinfo=timezone.utc).timestamp())
    f_dateFilter = (df['time'] >= start_ts) & (df['time'] <= end_ts)

    # Buy condition: close > ma1 AND close < ma2 AND f_dateFilter
    # Note: strategy.position_size check handled by tracking in_position
    buyCondition = (df['close'] > ma1) & (df['close'] < ma2) & f_dateFilter

    entries = []
    trade_num = 1
    in_position = False

    for i in range(len(df)):
        if pd.isna(ma1.iloc[i]) or pd.isna(ma2.iloc[i]):
            continue

        if buyCondition.iloc[i] and not in_position:
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])

            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

            trade_num += 1
            in_position = True

    return entries