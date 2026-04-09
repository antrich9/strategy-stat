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
    # Default parameters (NNFX Full System Tester defaults)
    # Baseline: EMA Cloud
    ema_fast_len = 8
    ema_slow_len = 21
    # Volume: Volume Oscillator
    vo_short_len = 5
    vo_long_len = 10
    # Confirmation 1: Zero Lag MACD
    zl_macd_fast = 12
    zl_macd_slow = 26
    zl_macd_signal = 9
    # Confirmation 2: TTMS (simplified placeholder, use baseline for demo)
    # Entry Mode: All Must Agree (default)

    # Compute Baseline (EMA Cloud)
    ema_fast = df['close'].ewm(span=ema_fast_len, adjust=False).mean()
    ema_slow = df['close'].ewm(span=ema_slow_len, adjust=False).mean()
    baseline_long = ema_fast > ema_slow
    baseline_short = ema_fast < ema_slow

    # Compute Volume Oscillator
    vol_short_ma = df['volume'].rolling(window=vo_short_len).mean()
    vol_long_ma = df['volume'].rolling(window=vo_long_len).mean()
    # Avoid division by zero
    vol_osc = (vol_short_ma - vol_long_ma) / (vol_long_ma + 1e-10) * 100
    volume_long = vol_osc > 0
    volume_short = vol_osc < 0

    # Compute Zero Lag MACD (using standard MACD approximation)
    macd_fast_ema = df['close'].ewm(span=zl_macd_fast, adjust=False).mean()
    macd_slow_ema = df['close'].ewm(span=zl_macd_slow, adjust=False).mean()
    macd_line = macd_fast_ema - macd_slow_ema
    macd_signal_line = macd_line.ewm(span=zl_macd_signal, adjust=False).mean()
    confirm1_long = macd_line > macd_signal_line
    confirm1_short = macd_line < macd_signal_line

    # Confirmation 2: TTMS placeholder (use baseline for demo, set to True for "All Must Agree")
    confirm2_long = pd.Series(True, index=df.index)
    confirm2_short = pd.Series(True, index=df.index)

    # Combine conditions based on "All Must Agree" mode
    long_condition = baseline_long & volume_long & confirm1_long & confirm2_long
    short_condition = baseline_short & volume_short & confirm1_short & confirm2_short

    # Fill NaNs with False
    long_condition = long_condition.fillna(False)
    short_condition = short_condition.fillna(False)

    # Detect crossovers (entry signals)
    # Entry when condition becomes true (current True, previous False)
    prev_long = long_condition.shift(1).fillna(False)
    prev_short = short_condition.shift(1).fillna(False)
    long_entry = long_condition & ~prev_long
    short_entry = short_condition & ~prev_short

    # Build entries list
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if long_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    return entries