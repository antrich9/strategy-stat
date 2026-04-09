import pandas as pd
import numpy as np
from datetime import datetime, timezone

def wma(series: pd.Series, length: int) -> pd.Series:
    """Weighted Moving Average"""
    def calc(row, idx, s, l):
        if idx < l - 1:
            return np.nan
        vals = s.iloc[idx - l + 1:idx + 1].values
        weights = np.arange(1, l + 1)
        return np.dot(vals, weights) / weights.sum()
    
    result = pd.Series(index=series.index, dtype=float)
    for i in range(len(series)):
        result.iloc[i] = calc(series.iloc[i], i, series, length)
    return result

def gd_t3(src: pd.Series, length: int, factor: float) -> pd.Series:
    """GD function for T3"""
    ema1 = src.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    return ema1 * (1 + factor) - ema2 * factor

def t3(src: pd.Series, length: int, factor: float) -> pd.Series:
    """T3 indicator"""
    t3_val = gd_t3(src, length, factor)
    t3_val = gd_t3(t3_val, length, factor)
    t3_val = gd_t3(t3_val, length, factor)
    return t3_val

def wilder_rsi(src: pd.Series, length: int) -> pd.Series:
    """Wilder RSI implementation"""
    delta = src.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Wilder ATR implementation"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/length, adjust=False).mean()
    return atr

def coppock_roc(src: pd.Series, length: int) -> pd.Series:
    """Rate of Change for Coppock"""
    return (src - src.shift(length)) / src.shift(length) * 100

def garma_volatility(high: pd.Series, low: pd.Series, close: pd.Series, open_: pd.Series, length: int) -> pd.Series:
    """Garman-Klass Volatility"""
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    gk[gk < 0] = 0
    gcv = np.sqrt(gk)
    return np.sqrt(gcv.rolling(length).mean())

def generate_entries(df: pd.DataFrame) -> list:
    """Generate entry signals from Pine Script strategy"""
    
    length_hull_ma = 9
    src_hull_ma = df['close']
    use_hull_ma = True
    use_color_hull_ma = True
    
    length_t3 = 5
    factor_t3 = 0.7
    src_t3 = df['close']
    use_t3 = True
    cross_t3 = True
    inverse_t3 = False
    highlight_movements_t3 = True
    
    length_coppock = 10
    long_roc_coppock = 14
    short_roc_coppock = 11
    src_coppock = df['close']
    length_coppock_ma = 10
    use_coppock_curve = True
    signal_logic_coppock = 'Zero line'
    
    length_gcv = 14
    gcv_ma_len = 50
    
    half_len = int(length_hull_ma / 2)
    sqrt_len = int(np.floor(np.sqrt(length_hull_ma)))
    
    hull_ma = wma(2 * wma(src_hull_ma, half_len) - wma(src_hull_ma, length_hull_ma), sqrt_len)
    
    t3_val = t3(src_t3, length_t3, factor_t3)
    
    roc_long = coppock_roc(src_coppock, long_roc_coppock)
    roc_short = coppock_roc(src_coppock, short_roc_coppock)
    coppock_raw = wma(roc_long + roc_short, length_coppock)
    coppock_ma = coppock_raw.ewm(span=length_coppock_ma, adjust=False).mean()
    
    gcv = garma_volatility(df['high'], df['low'], df['close'], df['open'], length_gcv)
    gcv_ma = gcv.rolling(gcv_ma_len).mean()
    gcv_filter = gcv > gcv_ma
    
    sig_hull_ma = (hull_ma > hull_ma.shift(1)).astype(int).replace(0, -1)
    t3_signals = (t3_val > t3_val.shift(1)).astype(int).replace(0, -1)
    
    if use_hull_ma:
        if use_color_hull_ma:
            signal_hull_long = (sig_hull_ma > 0) & (df['close'] > hull_ma)
            signal_hull_short = (sig_hull_ma < 0) & (df['close'] < hull_ma)
        else:
            signal_hull_long = df['close'] > hull_ma
            signal_hull_short = df['close'] < hull_ma
    else:
        signal_hull_long = pd.Series(True, index=df.index)
        signal_hull_short = pd.Series(True, index=df.index)
    
    if use_t3:
        if highlight_movements_t3:
            t3_signals_long = (t3_signals > 0) & (df['close'] > t3_val)
            t3_signals_short = (t3_signals < 0) & (df['close'] < t3_val)
        else:
            t3_signals_long = df['close'] > t3_val
            t3_signals_short = df['close'] < t3_val
    else:
        t3_signals_long = pd.Series(True, index=df.index)
        t3_signals_short = pd.Series(True, index=df.index)
    
    if cross_t3:
        t3_signals_long_cross = (~t3_signals_long.shift(1).fillna(False)) & t3_signals_long
        t3_signals_short_cross = (~t3_signals_short.shift(1).fillna(False)) & t3_signals_short
    else:
        t3_signals_long_cross = t3_signals_long
        t3_signals_short_cross = t3_signals_short
    
    if inverse_t3:
        t3_signals_long_final = ~t3_signals_long_cross
        t3_signals_short_final = ~t3_signals_short_cross
    else:
        t3_signals_long_final = t3_signals_long_cross
        t3_signals_short_final = t3_signals_short_cross
    
    if signal_logic_coppock == 'Zero line':
        coppock_long_cond = coppock_raw > 0
        coppock_short_cond = coppock_raw < 0
    else:
        coppock_long_cond = coppock_raw > coppock_ma
        coppock_short_cond = coppock_raw < coppock_ma
    
    if use_coppock_curve:
        coppock_long_final = coppock_long_cond
        coppock_short_final = coppock_short_cond
    else:
        coppock_long_final = pd.Series(True, index=df.index)
        coppock_short_final = pd.Series(True, index=df.index)
    
    entry_signal_long = gcv_filter & coppock_long_final & signal_hull_long & t3_signals_long_final
    entry_signal_short = gcv_filter & coppock_short_final & signal_hull_short & t3_signals_short_final
    
    entries = []
    trade_num = 1
    in_position = False
    
    for i in range(1, len(df)):
        if np.isnan(hull_ma.iloc[i]) or np.isnan(t3_val.iloc[i]) or np.isnan(coppock_raw.iloc[i]) or np.isnan(gcv.iloc[i]):
            continue
        
        if not in_position:
            if entry_signal_long.iloc[i]:
                entry_ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
                entry_price = float(df['close'].iloc[i])
                
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
            elif entry_signal_short.iloc[i]:
                entry_ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
                entry_price = float(df['close'].iloc[i])
                
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
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
    
    return entries