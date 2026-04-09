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
    # Extract close series
    close = df['close']

    # EMAs (Pine ta.ema)
    fastEMA = close.ewm(span=9, adjust=False).mean()
    slowEMA = close.ewm(span=18, adjust=False).mean()

    # Trend conditions
    isWeakBullishTrend = (close > slowEMA) & (fastEMA > slowEMA)
    isWeakBearishTrend = (close < slowEMA) & (fastEMA < slowEMA)

    # Crossover / crossunder (Pine ta.crossover, ta.crossunder)
    crossover_long = (close > fastEMA) & (close.shift(1) <= fastEMA.shift(1))
    crossunder_short = (close < fastEMA) & (close.shift(1) >= fastEMA.shift(1))

    # Entry signals
    long_condition = crossover_long & isWeakBullishTrend
    short_condition = crossunder_short & isWeakBearishTrend

    # Fill NaNs with False
    long_condition = long_condition.fillna(False)
    short_condition = short_condition.fillna(False)

    # Trade direction filter (default Both)
    trade_direction = "Both"

    entries = []
    trade_num = 1

    # Iterate bars (start at 1 because crossover uses prior bar)
    for i in range(1, len(df)):
        # Skip bars where required indicators are NaN
        if pd.isna(fastEMA.iloc[i]) or pd.isna(slowEMA.iloc[i]) or pd.isna(close.iloc[i]):
            continue

        ts = int(df['time'].iloc[i])
        entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = float(close.iloc[i])

        if trade_direction in ("Long", "Both") and long_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif trade_direction in ("Short", "Both") and short_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries