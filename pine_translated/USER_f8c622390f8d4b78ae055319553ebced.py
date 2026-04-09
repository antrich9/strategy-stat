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
    # ─── Parameters ────────────────────────────────────────────────────────
    length = 10
    adx_length = 14
    adx_smoothing = 14
    adx_threshold = 20.0
    adx_ma_length = 14

    close = df['close']
    high = df['high']
    low = df['low']

    # ─── McGinley Dynamic ───────────────────────────────────────────────────
    sma_series = close.rolling(length).mean()
    md = pd.Series(np.nan, index=close.index)
    for i in range(len(df)):
        if i == 0 or pd.isna(md.iloc[i - 1]):
            md.iloc[i] = sma_series.iloc[i] if not pd.isna(sma_series.iloc[i]) else np.nan
        else:
            prev_md = md.iloc[i - 1]
            price = close.iloc[i]
            md.iloc[i] = prev_md + (price - prev_md) / (length * (price / prev_md) ** 4)

    # ─── ADX + DI ──────────────────────────────────────────────────────────
    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    # Directional Movement
    high_diff = high.diff(1)
    low_diff = low.shift(1) - low

    plus_dm = pd.Series(
        np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0),
        index=high.index
    )
    minus_dm = pd.Series(
        np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0),
        index=low.index
    )

    # Wilder smoothed values
    alpha = 1.0 / adx_length
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()
    tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()

    # DI
    di_plus = 100.0 * (plus_dm_smooth / tr_smooth)
    di_minus = 100.0 * (minus_dm_smooth / tr_smooth)

    # DX
    dx = 100.0 * ((di_plus - di_minus).abs() / (di_plus + di_minus))

    # ADX (Wilder smoothed DX)
    adx_alpha = 1.0 / adx_smoothing
    adx = dx.ewm(alpha=adx_alpha, adjust=False).mean()

    # ADX MA
    adx_ma = adx.rolling(adx_ma_length).mean()

    # Confirmation: ADX above its MA and above threshold
    adx_confirm = (adx > adx_ma) & (adx > adx_threshold)

    # ─── Entry Conditions ──────────────────────────────────────────────────
    long_condition = (md > md.shift(1)) & (close > md) & adx_confirm & (di_plus > di_minus)
    short_condition = (md < md.shift(1)) & (close < md) & adx_confirm & (di_minus > di_plus)

    # ─── Generate Entries ─────────────────────────────────────────────────
    entries = []
    trade_num = 1
    in_position = False

    for i in range(len(df)):
        if not in_position:
            if long_condition.iloc[i]:
                entry_price = float(close.iloc[i])
                entry_ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
                in_position = True
            elif short_condition.iloc[i]:
                entry_price = float(close.iloc[i])
                entry