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
    # Work on a copy to avoid mutating the input
    df = df.copy()

    # ── Volatility Quality (VQ) parameters ──────────────────────────────────
    length_vq = 14          # lengthVQ
    smooth_len_vq = 5       # smoothLengthVQ

    # ── True Range and Directional Movement ─────────────────────────────────
    df['prev_close'] = df['close'].shift(1)
    df['truerange_vq'] = np.maximum(
        np.maximum(
            df['high'] - df['low'],
            np.abs(df['high'] - df['prev_close'])
        ),
        np.abs(df['low'] - df['prev_close'])
    )
    df['dm_vq'] = df['close'] - df['prev_close']

    # ── VQI (Volatility Quality Index) ───────────────────────────────────────
    df['sum_dm_vq'] = df['dm_vq'].rolling(window=length_vq, min_periods=length_vq).sum()
    df['sum_tr_vq'] = df['truerange_vq'].rolling(window=length_vq, min_periods=length_vq).sum()
    df['vqi_vq'] = df['sum_dm_vq'] / df['sum_tr_vq']
    # Guard against division by zero / inf
    df['vqi_vq'] = df['vqi_vq'].replace([np.inf, -np.inf], np.nan)

    # ── Smoothed VQ (Vqzla) ───────────────────────────────────────────────────
    df['vqzla_vq'] = df['vqi_vq'].rolling(window=smooth_len_vq, min_periods=smooth_len_vq).mean()

    # ── Entry signals ────────────────────────────────────────────────────────
    # crossVQ = true → use crossover / crossunder with the zero line
    df['crossover_vq'] = (df['vqzla_vq'] > 0) & (df['vqzla_vq'].shift(1) <= 0)
    df['crossunder_vq'] = (df['vqzla_vq'] < 0) & (df['vqzla_vq'].shift(1) >= 0)

    # long_condition = signalLongVQ (crossover), short_condition = signalShortVQ (crossunder)
    df['long_condition'] = df['crossover_vq']
    df['short_condition'] = df['crossunder_vq']

    # Ensure the indicator is valid (not NaN) before triggering
    valid = df['vqzla_vq'].notna()
    df['long_condition'] = df['long_condition'] & valid
    df['short_condition'] = df['short_condition'] & valid

    # ── Build entry list ──────────────────────────────────────────────────────
    entries = []
    trade_num = 1

    for i in range(len(df)):
        # Long entry (priority over short on same bar)
        if df['long_condition'].iloc[i]:
            ts = int(df['time'].iloc[i])
            price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
        # Short entry
        elif df['short_condition'].iloc[i]:
            ts = int(df['time'].iloc[i])
            price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1

    return entries