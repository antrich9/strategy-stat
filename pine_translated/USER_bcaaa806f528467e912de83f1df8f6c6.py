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
    # Parameters (default values from Pine Script)
    adx_length = 14
    adx_smoothing = 14
    adx_threshold = 25.0
    donch_length = 20  # not used for entry
    min_body_pct = 70
    atr_length = 100
    search_factor = 1.3
    ma_length = 20
    ma_length_fast = 8
    ma_reaction = 1
    # operation mode not needed for entry
    # filter type not needed for entry (always true)
    activate_green = True
    activate_red = True  # not used for long entries

    # ---- Helper: Wilder (exponential) smoothing ----
    def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(alpha=1.0 / period, adjust=False).mean()

    # ---- True Range ----
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ---- Directional Movement ----
    high_diff = df['high'] - df['high'].shift(1)
    low_diff = df['low'].shift(1) - df['low']

    plus_dm = pd.Series(
        np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0),
        index=df.index
    )

    # ---- Smoothed TR and DM ----
    smooth_tr = wilder_smooth(tr, adx_length)
    smooth_plus_dm = wilder_smooth(plus_dm, adx_length)
    smooth_minus_dm = wilder_smooth(minus_dm, adx_length)

    # ---- DI ----
    # avoid division by zero
    plus_di = 100.0 * smooth_plus_dm / smooth_tr
    minus_di = 100.0 * smooth_minus_dm / smooth_tr
    plus_di = plus_di.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    minus_di = minus_di.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # ---- DX ----
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # ---- ADX ----
    adx = wilder_smooth(dx, adx_smoothing).fillna(0.0)
    is_adx_strong = adx > adx_threshold

    # ---- ATR for Elephant Candles ----
    atr = wilder_smooth(tr, atr_length).fillna(0.0)

    # ---- Elephant Candle detection ----
    body = (df['close'] - df['open']).abs()
    ranges = df['high'] - df['low']
    body_pct = body / ranges * 100.0
    body_pct = body_pct.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    is_green = df['close'] > df['open']
    is_red = df['close'] < df['open']

    is_green_valid = is_green & (body_pct >= min_body_pct)
    is_red_valid = is_red & (body_pct >= min_body_pct)

    # body >= ATR[1] * searchFactor
    is_green_strong = is_green_valid & (body >= atr.shift(1) * search_factor)
    is_red_strong = is_red_valid & (body >= atr.shift(1) * search_factor)

    # ---- Moving Averages ----
    slow_ma = df['close'].rolling(window=ma_length).mean()
    fast_ma = df['close'].rolling(window=ma_length_fast).mean()

    # ---- Trend detection (rising/falling) ----
    # Reaction length = ma_reaction (default 1)
    def compute_trend(ma_series, reaction):
        trend = pd.Series(0, index=ma_series.index, dtype=float)
        # first 'reaction' bars remain 0
        for i in range(reaction, len(ma_series)):
            if ma_series.iloc[i] > ma_series.iloc[i - reaction]:
                trend.iloc[i] = 1.0
            elif ma_series.iloc[i] < ma_series.iloc[i - reaction]:
                trend.iloc[i] = -1.0
            else:
                trend.iloc[i] = trend.iloc[i - 1]
        return trend

    fast_trend = compute_trend(fast_ma, ma_reaction)
    # slow_trend not required for default long condition

    # ---- Trend conditions ----
    # Default bullish trend condition: DIRECCION MEDIA RAPIDA ALCISTA -> fast_trend > 0
    is_fast_bullish = (fast_trend > 0).fillna(False)

    # ---- Final green elephant candle ----
    final_green = is_green_strong & is_fast_bullish

    # ---- Entry condition ----
    entry_cond = final_green & is_adx_strong

    # ---- Generate entries ----
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if entry_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries