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
    # Strategy parameters
    fib_level = 0.7
    pivot_lookback = 10
    min_profit_pct = 1.0

    # Validate columns
    for col in ['time', 'open', 'high', 'low', 'close']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    n = len(df)
    if n == 0:
        return []

    # ---------- pivot high / low ----------
    left = pivot_lookback
    right = pivot_lookback

    # left side max/min (previous bars)
    left_max = df['high'].rolling(window=left).max().shift(1)
    left_min = df['low'].rolling(window=left).min().shift(1)

    # right side max/min (future bars) – compute on reversed series then reverse back
    rev_high = df['high'].iloc[::-1]
    rev_low = df['low'].iloc[::-1]

    right_max_rev = rev_high.rolling(window=right).max().shift(1)
    right_min_rev = rev_low.rolling(window=right).min().shift(1)

    right_max = right_max_rev.iloc[::-1]
    right_min = right_min_rev.iloc[::-1]

    # Pivot high: high > left_max and high > right_max
    pivot_high = df['high'].where((df['high'] > left_max) & (df['high'] > right_max))
    pivot_low = df['low'].where((df['low'] < left_min) & (df['low'] < right_min))

    # Previous bar low/high for retracement check
    prev_low = df['low'].shift(1)
    prev_high = df['high'].shift(1)

    # ---------- state variables ----------
    last_pivot_high = None
    last_pivot_low = None
    last_pivot_high_bar = -1
    last_pivot_low_bar = -1

    in_trade = False
    trade_counter = 0
    entries = []

    # ---------- bar‑by‑bar loop ----------
    for i in range(n):
        # Update most recent pivot high
        ph = pivot_high.iloc[i]
        if pd.notna(ph):
            last_pivot_high = float(ph)
            last_pivot_high_bar = i

        # Update most recent pivot low
        pl = pivot_low.iloc[i]
        if pd.notna(pl):
            last_pivot_low = float(pl)
            last_pivot_low_bar = i

        # Fibonacci retracement level
        if last_pivot_high is not None and last_pivot_low is not None:
            fib_price = last_pivot_high - (last_pivot_high - last_pivot_low) * fib_level
        else:
            fib_price = np.nan

        # Bullish candle
        bullish_candle = df['close'].iloc[i] > df['open'].iloc[i]

        # Retracement condition (previous bar)
        if i > 0 and pd.notna(fib_price):
            has_retraced = (prev_low.iloc[i] <= fib_price) and (prev_high.iloc[i] >= fib_price)
        else:
            has_retraced = False

        # Potential profit percentage
        if last_pivot_high is not None:
            potential_profit_pct = (last_pivot_high - df['close'].iloc[i]) / df['close'].iloc[i] * 100.0
        else:
            potential_profit_pct = np.nan

        # Reset in‑trade flag before evaluating entry
        if in_trade and (i > last_pivot_high_bar or i > last_pivot_low_bar):
            in_trade = False

        # Entry condition
        entry_cond = (
            bullish_candle
            and has_retraced
            and last_pivot_high is not None
            and last_pivot_low is not None
            and potential_profit_pct >= min_profit_pct
            and not in_trade
        )

        if entry_cond:
            trade_counter += 1
            entry_ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])

            entries.append({
                'trade_num': trade_counter,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

            in_trade = True

    return entries