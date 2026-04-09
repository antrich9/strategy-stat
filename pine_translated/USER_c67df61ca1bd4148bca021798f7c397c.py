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
    # Default NNFX system config (EMA Cloud baseline, Volume Oscillator, Zero Lag MACD, All 3 Must Agree)
    baselineType = "EMA Cloud"
    volumeType = "Volume Oscillator"
    confirmType = "Zero Lag MACD"
    entryMode = "All 3 Must Agree"
    emaF, emaS = 8, 21
    voSLen, voLLen, voThresh = 5, 10, 0.0
    zlF, zlS, zlSig = 12, 26, 9

    # Prepare indicators
    ema_fast = df['close'].ewm(span=emaF, adjust=False).mean()
    ema_slow = df['close'].ewm(span=emaS, adjust=False).mean()

    vol_short = df['volume'].rolling(voSLen).mean()
    vol_long = df['volume'].rolling(voLLen).mean()
    vo = (vol_short - vol_long) / vol_long
    vo_smooth = vo.ewm(span=voLLen, adjust=False).mean().fillna(0)

    zl_macd_raw = df['close'].ewm(span=zlF, adjust=False).mean() - df['close'].ewm(span=zlS, adjust=False).mean()
    zl_macd = zl_macd_raw.ewm(span=20, adjust=False).mean().fillna(0)
    zl_signal_raw = zl_macd.ewm(span=zlSig, adjust=False).mean()
    zl_signal = zl_signal_raw.ewm(span=20, adjust=False).mean().fillna(0)

    # Entry logic with three components
    max_len = max(emaS, voLLen, 20)
    entries = []
    trade_num = 1

    for i in range(max_len, len(df)):
        # Baseline: EMA Cloud direction
        baseline_long = ema_fast.iloc[i] > ema_slow.iloc[i]
        baseline_short = ema_fast.iloc[i] < ema_slow.iloc[i]

        # Volume: Volume Oscillator
        vol_confirm_long = vo_smooth.iloc[i] > voThresh
        vol_confirm_short = vo_smooth.iloc[i] < -voThresh

        # Confirmation: ZL MACD crossover/crossunder
        macd_above_prev = zl_macd.iloc[i] > zl_signal.iloc[i]
        macd_above_prev_ref = zl_macd.iloc[i-1] <= zl_signal.iloc[i-1]
        macd_below_prev = zl_macd.iloc[i] < zl_signal.iloc[i]
        macd_below_prev_ref = zl_macd.iloc[i-1] >= zl_signal.iloc[i-1]

        confirm_long = macd_above_prev and macd_above_prev_ref
        confirm_short = macd_below_prev and macd_below_prev_ref

        # Combined entry conditions based on entry mode
        if entryMode == "All 3 Must Agree":
            long_entry = baseline_long and vol_confirm_long and confirm_long
            short_entry = baseline_short and vol_confirm_short and confirm_short
        elif entryMode == "Baseline + Volume":
            long_entry = baseline_long and vol_confirm_long
            short_entry = baseline_short and vol_confirm_short
        elif entryMode == "Baseline + Confirmation":
            long_entry = baseline_long and confirm_long
            short_entry = baseline_short and confirm_short
        else:
            long_entry = baseline_long
            short_entry = baseline_short

        if long_entry:
            ts = int(df['time'].iloc[i])
            entry = {
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
            }
            entries.append(entry)
            trade_num += 1
        elif short_entry:
            ts = int(df['time'].iloc[i])
            entry = {
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
            }
            entries.append(entry)
            trade_num += 1

    return entries