import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    ts = df['time']

    t3_fast_length = 12
    t3_slow_length = 25
    t3_factor = 0.7
    adx_length = 14
    di_length = 14
    adx_threshold = 25.0
    bb_length = 20
    bb_mult = 2.0
    keltner_length = 20
    keltner_mult = 1.5
    entry_filter = 3
    trade_style = "Balanced"

    alpha = 2.0 / (t3_fast_length + 1)
    ema1 = close.ewm(span=t3_fast_length, adjust=False).mean()
    ema2 = ema1.ewm(span=t3_fast_length, adjust=False).mean()
    ema3 = ema2.ewm(span=t3_fast_length, adjust=False).mean()
    ema4 = ema3.ewm(span=t3_fast_length, adjust=False).mean()
    ema5 = ema4.ewm(span=t3_fast_length, adjust=False).mean()
    ema6 = ema5.ewm(span=t3_fast_length, adjust=False).mean()
    c1 = -t3_factor ** 3
    c2 = 3 * t3_factor ** 2 + 3 * t3_factor ** 3
    c3 = -6 * t3_factor ** 2 - 3 * t3_factor - 3 * t3_factor ** 3
    c4 = 1 + 3 * t3_factor + t3_factor ** 3 + 3 * t3_factor ** 2
    t3_fast = c1 * ema6 + c2 * ema5 + c3 * ema4 + c4 * ema3

    alpha_slow = 2.0 / (t3_slow_length + 1)
    ema1_s = close.ewm(span=t3_slow_length, adjust=False).mean()
    ema2_s = ema1_s.ewm(span=t3_slow_length, adjust=False).mean()
    ema3_s = ema2_s.ewm(span=t3_slow_length, adjust=False).mean()
    ema4_s = ema3_s.ewm(span=t3_slow_length, adjust=False).mean()
    ema5_s = ema4_s.ewm(span=t3_slow_length, adjust=False).mean()
    ema6_s = ema5_s.ewm(span=t3_slow_length, adjust=False).mean()
    t3_slow = c1 * ema6_s + c2 * ema5_s + c3 * ema4_s + c4 * ema3_s

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=close.index)
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.Series(np.maximum(np.maximum(tr1, tr2), tr3), index=close.index)
    tr_range = tr.ewm(alpha=1.0/di_length, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1.0/di_length, adjust=False).mean() / tr_range)
    minus_di = 100 * (minus_dm.ewm(alpha=1.0/di_length, adjust=False).mean() / tr_range)
    sum_di = plus_di + minus_di
    dx = 100 * (np.abs(plus_di - minus_di) / np.where(sum_di == 0, 1, sum_di))
    adx_value = dx.ewm(alpha=1.0/adx_length, adjust=False).mean()

    keltner_basis = close.ewm(span=keltner_length, adjust=False).mean()
    tr1_k = high - low
    tr2_k = np.abs(high - close.shift(1))
    tr3_k = np.abs(low - close.shift(1))
    tr_k = pd.Series(np.maximum(np.maximum(tr1_k, tr2_k), tr3_k), index=close.index)
    keltner_atr = tr_k.ewm(alpha=1.0/keltner_length, adjust=False).mean()
    keltner_upper = keltner_basis + keltner_mult * keltner_atr
    keltner_lower = keltner_basis - keltner_mult * keltner_atr

    bb_basis = close.rolling(bb_length).mean()
    bb_std = close.rolling(bb_length).std()
    bb_upper = bb_basis + bb_mult * bb_std
    bb_lower = bb_basis - bb_mult * bb_std

    src_price = (high + low) / 2

    long_stop = src_price - 3.0 * keltner_atr
    long_stop_prev = long_stop.shift(1).fillna(long_stop)
    long_stop = pd.Series(np.where(long_stop > 0, long_stop, long_stop_prev), index=close.index)
    long_stop = pd.Series(np.where(long_stop > 0, np.maximum(long_stop, long_stop_prev), long_stop_prev), index=close.index)

    short_stop = src_price + 3.0 * keltner_atr
    short_stop_prev = short_stop.shift(1).fillna(short_stop)
    short_stop = pd.Series(np.where(short_stop > 0, short_stop, short_stop_prev), index=close.index)
    short_stop = pd.Series(np.where(short_stop > 0, np.minimum(short_stop, short_stop_prev), short_stop_prev), index=close.index)

    high_shifted = high.shift(1).fillna(high)
    low_shifted = low.shift(1).fillna(low)

    bars_since_last_entry = 9999
    in_long_position = False
    in_short_position = False

    raw_long_signal = pd.Series(False, index=close.index)
    raw_short_signal = pd.Series(False, index=close.index)

    if trade_style == "Strict":
        for i in range(1, len(close)):
            cross_up = t3_fast.iloc[i] > t3_slow.iloc[i] and t3_fast.iloc[i-1] <= t3_slow.iloc[i-1]
            cross_down = t3_fast.iloc[i] < t3_slow.iloc[i] and t3_fast.iloc[i-1] >= t3_slow.iloc[i-1]
            bullish_breakout = bb_upper.iloc[i] < keltner_upper.iloc[i] and bb_upper.iloc[i-1] >= keltner_upper.iloc[i-1]
            bearish_breakout = bb_lower.iloc[i] > keltner_lower.iloc[i] and bb_lower.iloc[i-1] <= keltner_lower.iloc[i-1]
            raw_long_signal.iloc[i] = cross_up and close.iloc[i] > t3_slow.iloc[i] and bullish_breakout and adx_value.iloc[i] > adx_threshold
            raw_short_signal.iloc[i] = cross_down and close.iloc[i] < t3_slow.iloc[i] and bearish_breakout and adx_value.iloc[i] > adx_threshold
    elif trade_style == "Balanced":
        raw_long_signal = (t3_fast > t3_slow) & (close > t3_slow) & (bb_upper < keltner_upper) & (adx_value > adx_threshold)
        raw_short_signal = (t3_fast < t3_slow) & (close < t3_slow) & (bb_lower > keltner_lower) & (adx_value > adx_threshold)
    elif trade_style == "Scalper":
        raw_long_signal = (close > t3_slow) & (bb_upper < keltner_upper) & (adx_value > adx_threshold)
        raw_short_signal = (close < t3_slow) & (bb_lower > keltner_lower) & (adx_value > adx_threshold)
    elif trade_style == "Exits Only":
        raw_long_signal = pd.Series(False, index=close.index)
        raw_short_signal = pd.Series(False, index=close.index)
    else:
        for i in range(1, len(close)):
            cross_up = t3_fast.iloc[i] > t3_slow.iloc[i] and t3_fast.iloc[i-1] <= t3_slow.iloc[i-1]
            cross_down = t3_fast.iloc[i] < t3_slow.iloc[i] and t3_fast.iloc[i-1] >= t3_slow.iloc[i-1]
            bullish_breakout = bb_upper.iloc[i] < keltner_upper.iloc[i] and bb_upper.iloc[i-1] >= keltner_upper.iloc[i-1]
            bearish_breakout = bb_lower.iloc[i] > keltner_lower.iloc[i] and bb_lower.iloc[i-1] <= keltner_lower.iloc[i-1]
            raw_long_signal.iloc[i] = cross_up and bullish_breakout and adx_value.iloc[i] > adx_threshold
            raw_short_signal.iloc[i] = cross_down and bearish_breakout and adx_value.iloc[i] > adx_threshold

    entries = []
    trade_num = 0
    max_len = len(close)
    warmup = max(t3_slow_length, di_length + adx_length, keltner_length, bb_length, entry_filter)

    for i in range(warmup, max_len):
        if pd.isna(t3_fast.iloc[i]) or pd.isna(t3_slow.iloc[i]) or pd.isna(adx_value.iloc[i]):
            continue
        close_price = close.iloc[i]
        ts_val = ts.iloc[i]
        entry_time = datetime.fromtimestamp(ts_val, tz=timezone.utc).isoformat()

        if trade_style != "Exits Only" and raw_long_signal.iloc[i] and bars_since_last_entry >= entry_filter and not in_long_position:
            trade_num += 1
            entries.append({'trade_num': trade_num, 'direction': 'long', 'entry_ts': int(ts_val), 'entry_time': entry_time, 'entry_price_guess': float(close_price), 'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0, 'raw_price_a': float(close_price), 'raw_price_b': float(close_price)})
            in_long_position = True
            in_short_position = False
            bars_since_last_entry = 0
        elif trade_style != "Exits Only" and raw_short_signal.iloc[i] and bars_since_last_entry >= entry_filter and not in_short_position:
            trade_num += 1
            entries.append({'trade_num': trade_num, 'direction': 'short', 'entry_ts': int(ts_val), 'entry_time': entry_time, 'entry_price_guess': float(close_price), 'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0, 'raw_price_a': float(close_price), 'raw_price_b': float(close_price)})
            in_short_position = True
            in_long_position = False
            bars_since_last_entry = 0

        bars_since_last_entry += 1

    return entries