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
    if len(df) < 5:
        return []

    entries = []
    trade_num = 1

    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    time = df['time']

    ema_200 = close.ewm(span=200, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()

    is_up = close > open_
    is_down = close < open_

    is_ob_up = is_down.shift(1) & is_up & (close > high.shift(1))
    is_ob_down = is_up.shift(1) & is_down & (close < low.shift(1))

    is_fvg_up = low > high.shift(2)
    is_fvg_down = high < low.shift(2)

    stacked_bullish = is_ob_up & is_fvg_up
    stacked_bearish = is_ob_down & is_fvg_down

    pdh = high.shift(1).rolling(window=2, min_periods=2).max().shift(1)
    pdl = low.shift(1).rolling(window=2, min_periods=2).min().shift(1)

    flagpdh = close > pdh
    flagpdl = close < pdl

    ema_200_bullish = ema_200 > ema_200.shift(1)
    ema_200_bearish = ema_200 < ema_200.shift(1)
    ema_50_above_200 = ema_50 > ema_200
    ema_50_below_200 = ema_50 < ema_200

    valid = ~(pd.isna(ema_200) | pd.isna(ema_50) | pd.isna(pdh) | pd.isna(pdl))

    bull_condition = flagpdh & stacked_bullish & ema_200_bullish & ema_50_above_200 & valid
    bear_condition = flagpdl & stacked_bearish & ema_200_bearish & ema_50_below_200 & valid

    for i in range(5, len(df)):
        if bull_condition.iloc[i]:
            fib_base = low.iloc[i]
            fib_618 = fib_base * 1.0618
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(time.iloc[i]),
                'entry_time': datetime.fromtimestamp(time.iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(fib_618),
                'raw_price_b': float(fib_618)
            })
            trade_num += 1

        if bear_condition.iloc[i]:
            fib_base = high.iloc[i]
            fib_618 = fib_base * 0.9382
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(time.iloc[i]),
                'entry_time': datetime.fromtimestamp(time.iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(fib_618),
                'raw_price_b': float(fib_618)
            })
            trade_num += 1

    return entries