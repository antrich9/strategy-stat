import pandas as pd
import numpy as np
from datetime import datetime, timezone

def wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def hull_ma(src: pd.Series, length: int) -> pd.Series:
    half_len = int(length / 2)
    hull_ma_val = 2 * wma(src, half_len) - wma(src, length)
    sqrt_len = int(np.sqrt(length))
    return wma(hull_ma_val, sqrt_len)

def calculate_t3(src: pd.Series, length: int, factor: float) -> pd.Series:
    ema1 = src.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    return ema1 * (1 + factor) - ema2 * factor

def wilder_rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    open_col = df['open']
    high = df['high']
    low = df['low']
    length_hull_ma = 9
    length_t3 = 5
    factor_t3 = 0.7
    ma_length_stiffness = 100
    stiff_length = 60
    stiff_smooth = 3
    threshold_stiffness = 90
    
    hull_ma_series = hull_ma(close, length_hull_ma)
    hull_ma_prev = hull_ma_series.shift(1)
    sig_hull_ma = (hull_ma_series > hull_ma_prev).astype(int) - (hull_ma_series < hull_ma_prev).astype(int)
    
    t3_temp = calculate_t3(close, length_t3, factor_t3)
    t3 = calculate_t3(t3_temp, length_t3, factor_t3)
    t3 = calculate_t3(t3, length_t3, factor_t3)
    t3_prev = t3.shift(1)
    t3_signals = (t3 > t3_prev).astype(int) - (t3 < t3_prev).astype(int)
    
    sma_stiff = close.rolling(window=ma_length_stiffness).mean()
    stdev_stiff = close.rolling(window=ma_length_stiffness).std()
    bound_stiffness = sma_stiff - 0.2 * stdev_stiff
    close_above_bound = (close > bound_stiffness).astype(int)
    sum_above_stiff = close_above_bound.rolling(window=stiff_length).sum()
    stiffness = (sum_above_stiff * 100 / stiff_length).ewm(span=stiff_smooth, adjust=False).mean()
    
    use_hull_ma = True
    usecolor_hull_ma = True
    use_t3 = True
    cross_t3 = True
    inverse_t3 = False
    highlight_movements_t3 = True
    use_stiffness = False
    
    signal_hull_ma_long = close > hull_ma_series if use_hull_ma else pd.Series(True, index=df.index)
    signal_hull_ma_short = close < hull_ma_series if use_hull_ma else pd.Series(True, index=df.index)
    
    basic_long_cond = (t3_signals > 0) & (close > t3)
    basic_short_cond = (t3_signals < 0) & (close < t3)
    
    t3_signals_long = basic_long_cond if use_t3 else (close > t3 if use_t3 else pd.Series(True, index=df.index))
    t3_signals_short = basic_short_cond if use_t3 else (close < t3 if use_t3 else pd.Series(True, index=df.index))
    
    t3_signals_long_prev = t3_signals_long.shift(1).fillna(0)
    t3_signals_short_prev = t3_signals_short.shift(1).fillna(0)
    
    t3_signals_long_cross = (t3_signals_long == 1) & (t3_signals_long_prev <= 0)
    t3_signals_short_cross = (t3_signals_short == -1) & (t3_signals_short_prev >= 0)
    
    t3_signals_long_final = (~t3_signals_long_cross) if inverse_t3 else t3_signals_long_cross
    t3_signals_short_final = (~t3_signals_short_cross) if inverse_t3 else t3_signals_short_cross
    
    long_entry_cond = t3_signals_long_final & (stiffness > threshold_stiffness)
    short_entry_cond = t3_signals_short_final & (stiffness < threshold_stiffness)
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if np.isnan(hull_ma_series.iloc[i]) or np.isnan(t3.iloc[i]) or np.isnan(stiffness.iloc[i]):
            continue
        
        if long_entry_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        
        if short_entry_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
    
    return entries