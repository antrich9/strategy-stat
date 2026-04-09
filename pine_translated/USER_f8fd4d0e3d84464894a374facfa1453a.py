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
    # Ensure chronological order
    df = df.sort_values('time').reset_index(drop=True)

    # Strategy parameters
    length = 15
    inpT3Period = 5
    Hot = 0.7
    inplevels = 100

    # Highest high and lowest low over the lookback period
    high_max = df['high'].rolling(length).max()
    low_min = df['low'].rolling(length).min()

    # Shifted versions for the power calculations
    low_min_shifted = df['low'].shift(length).rolling(length).min()
    high_max_shifted = df['high'].shift(length).rolling(length).max()

    # Buy and Sell Power
    buyPower = high_max - low_min_shifted
    sellPower = high_max_shifted - low_min

    # Trend Trigger Factor (TTF)
    denom = 0.5 * (buyPower + sellPower)
    ttf = pd.Series(
        np.where(denom == 0, np.nan, 100 * (buyPower - sellPower) / denom),
        index=df.index
    )

    # T3 smoothing using exponential moving averages
    Period = max(inpT3Period, 1)
    e1 = ttf.ewm(span=Period, adjust=False).mean()
    e2 = e1.ewm(span=Period, adjust=False).mean()
    e3 = e2.ewm(span=Period, adjust=False).mean()
    e4 = e3.ewm(span=Period, adjust=False).mean()
    e5 = e4.ewm(span=Period, adjust=False).mean()
    e6 = e5.ewm(span=Period, adjust=False).mean()

    b = Hot
    b2 = b * b
    b3 = b2 * b

    c1 = -b3
    c2 = 3 * b2 + 3 * b3
    c3 = -6 * b2 - 3 * b - 3 * b3
    c4 = 1 + 3 * b + b3 + 3 * b2

    avg = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    # Determine color for each bar
    def _color(val):
        if pd.isna(val):
            return None
        if val >= inplevels:
            return "blue"
        if val <= -inplevels:
            return "red"
        return "gray"

    ttf_color = avg.apply(_color)

    entries = []
    trade_num = 1
    prev_color = None

    for i in range(1, len(df)):
        curr_color = ttf_color.iloc[i]
        if curr_color is None:
            prev_color = None
            continue

        if prev_color is not None:
            # Bullish condition: color turns red
            if prev_color != "red" and curr_color == "red":
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

            # Bearish condition: color turns blue
            elif prev_color != "blue" and curr_color == "blue":
                entry_ts = int(df['time'].iloc[i])
                entry_price = float(df['close'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
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

        prev_color = curr_color

    return entries