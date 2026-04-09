import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Ensure numeric types
    df = df.copy()
    df['time'] = df['time'].astype(int)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')

    # Parameters (matching Pine script)
    ema_length = 200
    supertrend_period = 10
    supertrend_multiplier = 3
    atr_length = 14
    adx_threshold = 20
    dmi_period = 14

    # ----- EMA -----
    ema = df['close'].ewm(span=ema_length, adjust=False).mean()

    # ----- True Range & ATR -----
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_length, adjust=False).mean()

    # ----- Supertrend -----
    period_st = supertrend_period
    multiplier_st = supertrend_multiplier
    atr_st = tr.ewm(alpha=1/period_st, adjust=False).mean()
    mat = (df['high'] + df['low']) / 2
    basic_upper = mat + multiplier_st * atr_st
    basic_lower = mat - multiplier_st * atr_st

    n = len(df)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    direction = np.full(n, np.nan)   # 1 = bullish, -1 = bearish

    # initial bar
    final_upper[0] = basic_upper.iloc[0]
    final_lower[0] = basic_lower.iloc[0]
    direction[0] = 1  # start bullish

    for i in range(1, n):
        prev_close = df['close'].iloc[i - 1]
        curr_upper = basic_upper.iloc[i]
        curr_lower = basic_lower.iloc[i]

        if prev_close > final_upper[i - 1]:
            direction[i] = -1
            final_upper[i] = curr_upper
            final_lower[i] = min(curr_lower, final_lower[i - 1])
        elif prev_close < final_lower[i - 1]:
            direction[i] = 1
            final_upper[i] = max(curr_upper, final_upper[i - 1])
            final_lower[i] = curr_lower
        else:
            direction[i] = direction[i - 1]
            if direction[i] == 1:
                final_upper[i] = max(curr_upper, final_upper[i - 1])
                final_lower[i] = curr_lower
            else:
                final_upper[i] = curr_upper
                final_lower[i] = min(curr_lower, final_lower[i - 1])

    supertrend_direction = pd.Series(direction, index=df.index)

    # ----- ADX (DMI) -----
    high_diff = df['high'] - df['high'].shift(1)
    low_diff = df['low'].shift(1) - df['low']

    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)

    smoothed_plus_dm = plus_dm.ewm(alpha=1/dmi_period, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1/dmi_period, adjust=False).mean()

    plus_di = 100 * smoothed_plus_dm / atr
    minus_di = 100 * smoothed_minus_dm / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/dmi_period, adjust=False).mean()

    # ----- Entry Conditions -----
    close_series = df['close']
    is_ema_bullish = close_series > ema
    is_ema_bearish = close_series < ema
    is_supertrend_bullish = supertrend_direction == 1
    is_supertrend_bearish = supertrend_direction == -1
    is_strong_trend = adx > adx_threshold

    long_entry = is_ema_bullish & is_supertrend_bullish & is_strong_trend
    short_entry = is_ema_bearish & is_supertrend_bearish & is_strong_trend

    # ----- Build Entry List -----
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if long_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            price = close_series.iloc[i]
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
        elif short_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            price = close_series.iloc[i]
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

    return entries