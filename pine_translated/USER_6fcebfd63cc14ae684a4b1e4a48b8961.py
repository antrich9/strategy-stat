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
    # Determine pip based on median price (JPY pairs have price > 10)
    median_price = df['close'].median()
    pip = 0.02 if median_price > 10 else 0.0002
    offset = pip * 2

    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']

    # EMAs
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    # Doji detection
    body = (close - open_).abs()
    rng = high - low
    isDoji = (rng > 0) & (body / rng <= 0.30)

    # EMA trend conditions
    emaBull = (ema8 > ema20) & (ema20 > ema50)
    emaBear = (ema8 < ema20) & (ema20 < ema50)

    # Price location within bar
    closeInUpper33 = (close - low) / rng >= 0.67
    closeInLower33 = (high - close) / rng >= 0.67
    bodyInUpper33 = (open_ >= low + rng * 0.67) & (close >= low + rng * 0.67)
    bodyInLower33 = (open_ <= low + rng * 0.33) & (close <= low + rng * 0.33)

    # Signal conditions
    bullSignal = emaBull & isDoji & (low <= ema8) & closeInUpper33 & bodyInUpper33
    bearSignal = emaBear & isDoji & (high >= ema8) & closeInLower33 & bodyInLower33

    bullOnly = bullSignal & ~bearSignal
    bearOnly = bearSignal & ~bullSignal

    entries = []
    trade_num = 1
    pending = []  # each element: {'entry_price': float, 'direction': str, 'signal_bar': int}

    n = len(df)
    for i in range(n):
        # Process pending order for this bar, if any
        if pending:
            order = pending[0]
            entry_price = order['entry_price']
            direction = order['direction']
            sig_bar = order['signal_bar']

            # The entry bar is the bar immediately after the signal bar
            if i == sig_bar + 1:
                # Check if the stop price was hit
                if direction == 'long' and high.iloc[i] >= entry_price:
                    ts = int(df['time'].iloc[i])
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': ts,
                        'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                        'entry_price_guess': entry_price,
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': entry_price,
                        'raw_price_b': entry_price
                    })
                    trade_num += 1
                elif direction == 'short' and low.iloc[i] <= entry_price:
                    ts = int(df['time'].iloc[i])
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': ts,
                        'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                        'entry_price_guess': entry_price,
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': entry_price,
                        'raw_price_b': entry_price
                    })
                    trade_num += 1
                # Remove processed pending order after the entry bar
                pending.pop(0)
            elif i > sig_bar + 1:
                # Defensive removal if entry bar was missed
                pending.pop(0)

        # Create new pending orders on signal bars (ignore position size)
        if bullOnly.iloc[i]:
            entry_price = high.iloc[i] + offset
            pending.append({'entry_price': entry_price, 'direction': 'long', 'signal_bar': i})
        elif bearOnly.iloc[i]:
            entry_price = low.iloc[i] - offset
            pending.append({'entry_price': entry_price, 'direction': 'short', 'signal_bar': i})

    return entries