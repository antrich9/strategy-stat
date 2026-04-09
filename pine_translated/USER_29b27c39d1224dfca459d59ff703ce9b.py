import pandas as pd
import numpy as np
from datetime import datetime, timezone

def weighted_rolling_mean(data: pd.Series, window: int) -> pd.Series:
    """Calculate Weighted Moving Average."""
    weights = np.arange(1, window + 1)
    weight_sum = weights.sum()
    result = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        start_idx = i - window + 1
        if start_idx >= 0:
            window_data = data.iloc[start_idx:i + 1]
            result.iloc[i] = np.sum(window_data * weights) / weight_sum
    return result

def calculate_hull_ma(close, length):
    half_length = length // 2
    sqrt_length = int(np.sqrt(length))
    wma1 = weighted_rolling_mean(close, half_length)
    wma2 = weighted_rolling_mean(close, length)
    hull_ma = 2 * wma1 - wma2
    hull_ma = weighted_rolling_mean(hull_ma, sqrt_length)
    return hull_ma

def calculate_t3(close, length, factor):
    ema1 = close.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    ema4 = ema3.ewm(span=length, adjust=False).mean()
    ema5 = ema4.ewm(span=length, adjust=False).mean()
    ema6 = ema5.ewm(span=length, adjust=False).mean()
    return ema1 * (1 + factor) + ema6 * factor

def calculate_wilder_rsi(data, length):
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(index=data.index, dtype=float)
    avg_loss = pd.Series(index=data.index, dtype=float)
    avg_gain.iloc[length] = gain.iloc[1:length+1].mean()
    avg_loss.iloc[length] = loss.iloc[1:length+1].mean()
    for i in range(length + 1, len(data)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (length - 1) + gain.iloc[i]) / length
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (length - 1) + loss.iloc[i]) / length
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stc(close, length, fast_length, slow_length, factor):
    ema_fast = close.ewm(span=fast_length, adjust=False).mean()
    ema_slow = close.ewm(span=slow_length, adjust=False).mean()
    macd = ema_fast - ema_slow
    k = calculate_wilder_rsi(macd, length)
    d = k.ewm(span=3, adjust=False).mean()
    kd = k - d
    numerator = np.abs(kd)
    denom = numerator.rolling(length, min_periods=length).max()
    denom = denom.replace(0, np.nan)
    fract = numerator / denom * 100
    fract = fract.fillna(method='ffill')
    v2 = fract.ewm(alpha=factor, adjust=False).mean()
    return 100 * v2

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    hull_ma = calculate_hull_ma(close, 9)
    hull_ma_prev = hull_ma.shift(1)
    stc = calculate_stc(close, 10, 23, 50, 0.5)
    t3 = calculate_t3(close, 5, 0.7)
    t3_prev = t3.shift(1)
    signal_hull_long = (hull_ma > hull_ma_prev) & (close > hull_ma)
    t3_above_prev = t3 > t3_prev
    t3_signals_long = t3_above_prev & (close > t3)
    t3_signals_long_prev = t3_signals_long.shift(1).fillna(False).astype(bool)
    t3_cross = (~t3_signals_long_prev) & t3_signals_long
    t3_long_final = t3_cross
    stc_threshold = 25
    entry_condition = signal_hull_long & t3_long_final & (stc > stc_threshold)
    entries = []
    trade_num = 1
    for idx in range(1, len(df)):
        if entry_condition.iloc[idx]:
            ts = df['time'].iloc[idx]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[idx]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[idx]),
                'raw_price_b': float(close.iloc[idx])
            })
            trade_num += 1
    return entries