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
    # Basic validation
    required_cols = {'time', 'close'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Default confirmation indicator parameters (Zero Lag MACD)
    zl_fast = 12
    zl_slow = 26
    zl_signal = 9

    # Zero‑Lag EMA helper: ZLEMA = 2*EMA(src,len) - EMA(EMA(src,len),len)
    def zlema(series: pd.Series, length: int) -> pd.Series:
        ema1 = series.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return 2 * ema1 - ema2

    close_series = df['close']

    # Compute Zero‑Lag MACD components
    fast_zlema = zlema(close_series, zl_fast)
    slow_zlema = zlema(close_series, zl_slow)
    macd_line = fast_zlema - slow_zlema
    signal_line = macd_line.ewm(span=zl_signal, adjust=False).mean()

    # Previous bar values for crossover / crossunder detection
    prev_macd = macd_line.shift(1)
    prev_signal = signal_line.shift(1)

    # Ensure we have valid (non‑NaN) indicator values
    valid = macd_line.notna() & signal_line.notna()

    # Entry conditions
    long_cond = valid & (macd_line > signal_line) & (prev_macd <= prev_signal)
    short_cond = valid & (macd_line < signal_line) & (prev_macd >= prev_signal)

    # Build the list of entry records
    entries = []
    trade_num = 1

    for i in range(1, len(df)):
        if long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1

    return entries