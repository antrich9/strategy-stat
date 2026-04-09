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
    # Default parameters mirroring the Pine Script defaults
    # Baseline: EMA Cloud
    ema_fast_len = 8
    ema_slow_len = 21

    # Volume Oscillator
    vo_short_len = 5
    vo_long_len = 10
    vo_threshold = 0.0

    # Zero Lag MACD (Confirmation 1)
    zl_fast_len = 12
    zl_slow_len = 26
    zl_signal_len = 9

    # TTMS (Confirmation 2) – fast vs slow SMA difference
    ttms_fast_len = 20
    ttms_slow_len = 50

    # Entry mode – default "All Must Agree"
    entry_mode = "All Must Agree"

    # ---- Indicator calculations -------------------------------------------------
    # Baseline: EMA Cloud – crossover of fast EMA over slow EMA
    ema_fast = df['close'].ewm(span=ema_fast_len, adjust=False).mean()
    ema_slow = df['close'].ewm(span=ema_slow_len, adjust=False).mean()
    baseline_long = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
    baseline_short = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))

    # Volume Oscillator – short EMA of volume minus long EMA of volume
    vol_ema_short = df['volume'].ewm(span=vo_short_len, adjust=False).mean()
    vol_ema_long = df['volume'].ewm(span=vo_long_len, adjust=False).mean()
    vol_osc = vol_ema_short - vol_ema_long
    volume_long = (vol_osc > vo_threshold) & (vol_osc.shift(1) <= vo_threshold)
    volume_short = (vol_osc < -vo_threshold) & (vol_osc.shift(1) >= -vo_threshold)

    # Zero Lag MACD (Confirmation 1) – MACD line vs signal line crossover
    ema_zl_fast = df['close'].ewm(span=zl_fast_len, adjust=False).mean()
    ema_zl_slow = df['close'].ewm(span=zl_slow_len, adjust=False).mean()
    macd_line = ema_zl_fast - ema_zl_slow
    signal_line = macd_line.ewm(span=zl_signal_len, adjust=False).mean()
    confirm1_long = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    confirm1_short = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

    # TTMS (Confirmation 2) – fast SMA vs slow SMA crossover
    sma_fast = df['close'].rolling(window=ttms_fast_len).mean()
    sma_slow = df['close'].rolling(window=ttms_slow_len).mean()
    ttms_val = sma_fast - sma_slow
    confirm2_long = (ttms_val > 0) & (ttms_val.shift(1) <= 0)
    confirm2_short = (ttms_val < 0) & (ttms_val.shift(1) >= 0)

    # ---- Combine according to entry mode ----------------------------------------
    if entry_mode == "All Must Agree":
        long_cond = baseline_long & volume_long & confirm1_long & confirm2_long
        short_cond = baseline_short & volume_short & confirm1_short & confirm2_short
    elif entry_mode == "Baseline + Volume":
        long_cond = baseline_long & volume_long
        short_cond = baseline_short & volume_short
    elif entry_mode == "Baseline + C1":
        long_cond = baseline_long & confirm1_long
        short_cond = baseline_short & confirm1_short
    elif entry_mode == "Baseline + C2":
        long_cond = baseline_long & confirm2_long
        short_cond = baseline_short & confirm2_short
    elif entry_mode == "Baseline + C1 or C2":
        long_cond = baseline_long & (confirm1_long | confirm2_long)
        short_cond = baseline_short & (confirm1_short | confirm2_short)
    elif entry_mode == "Baseline Only":
        long_cond = baseline_long
        short_cond = baseline_short
    else:
        long_cond = pd.Series([False] * len(df))
        short_cond = pd.Series([False] * len(df))

    # ---- Build entry list --------------------------------------------------------
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
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
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1