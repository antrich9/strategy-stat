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
    close = df['close']
    ema_short = close.ewm(span=10, adjust=False).mean()
    ema_long = close.ewm(span=20, adjust=False).mean()
    htf_ema = close.ewm(span=50, adjust=False).mean()

    # Wilder RSI
    rsi_len = 14
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / rsi_len, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_len, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Entry conditions
    cond_long = (
        (ema_short > ema_long) &
        (ema_short.shift(1) <= ema_long.shift(1)) &
        (close > htf_ema) &
        (rsi > 50)
    )
    cond_short = (
        (ema_short < ema_long) &
        (ema_short.shift(1) >= ema_long.shift(1)) &
        (rsi < 50) &
        (close < htf_ema)
    )

    entries = []
    trade_num = 1
    for i in range(len(df)):
        if (
            pd.isna(ema_short.iloc[i]) or
            pd.isna(ema_long.iloc[i]) or
            pd.isna(rsi.iloc[i]) or
            pd.isna(htf_ema.iloc[i])
        ):
            continue

        if cond_long.iloc[i]:
            ts = int(df.iloc[i]['time'])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df.iloc[i]['close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df.iloc[i]['close']),
                'raw_price_b': float(df.iloc[i]['close'])
            })
            trade_num += 1
        elif cond_short.iloc[i]:
            ts = int(df.iloc[i]['time'])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df.iloc[i]['close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df.iloc[i]['close']),
                'raw_price_b': float(df.iloc[i]['close'])
            })
            trade_num += 1

    return entries