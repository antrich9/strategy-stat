import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Ensure columns are present
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Daily EMAs (assuming daily bars)
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    
    # Volume indicator
    sma50_vol = volume.rolling(50).mean()
    nVolume = volume / sma50_vol * 100
    hv = 150
    
    # Hull MA
    length_hull = 9
    half_hull = length_hull // 2
    def wma(series, window):
        weights = np.arange(1, window + 1)
        sum_w = window * (window + 1) // 2
        def weighted_avg(x):
            return np.dot(x, weights) / sum_w
        return series.rolling(window).apply(weighted_avg, raw=True)
    
    wma_half = wma(close, half_hull)
    wma_full = wma(close, length_hull)
    sqrt_hull = int(np.sqrt(length_hull))
    hullma = wma(2 * wma_half - wma_full, sqrt_hull)
    
    sig_hull = np.where(hullma > hullma.shift(1), 1, -1)
    sig_hull_series = pd.Series(sig_hull, index=close.index)
    signal_hull_long = (sig_hull_series == 1) & (close > hullma)
    
    # T3
    length_t3 = 5
    factor_t3 = 0.7
    ema1 = close.ewm(span=length_t3, adjust=False).mean()
    ema2 = ema1.ewm(span=length_t3, adjust=False).mean()
    gd1 = ema1 * (1 + factor_t3) - ema2 * factor_t3
    
    ema3 = gd1.ewm(span=length_t3, adjust=False).mean()
    ema4 = ema3.ewm(span=length_t3, adjust=False).mean()
    gd2 = ema3 * (1 + factor_t3) - ema4 * factor_t3
    
    ema5 = gd2.ewm(span=length_t3, adjust=False).mean()
    ema6 = ema5.ewm(span=length_t3, adjust=False).mean()
    t3 = ema5 * (1 + factor_t3) - ema6 * factor_t3
    
    t3_signals = np.where(t3 > t3.shift(1), 1, -1)
    t3_signals_series = pd.Series(t3_signals, index=close.index)
    basic_long_t3 = (t3_signals_series == 1) & (close > t3)
    # useT3 true, highlightMovements true => t3SignalsLong = basicLong
    t3_signals_long = basic_long_t3
    # cross condition
    t3_signals_long_cross = t3_signals_long & ~t3_signals_long.shift(1).fillna(False)
    
    # SuperTrend (trend1)
    period_st = 10
    multiplier_st = 3.0
    hl2 = (high + low) / 2
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    atr = tr.ewm(alpha=1/period_st, adjust=False).mean()
    up2_raw = hl2 - multiplier_st * atr
    dn2_raw = hl2 + multiplier_st * atr
    
    up2 = pd.Series(np.nan, index=close.index)
    dn2 = pd.Series(np.nan, index=close.index)
    up2.iloc[0] = up2_raw.iloc[0]
    dn2.iloc[0] = dn2_raw.iloc[0]
    for i in range(1, len(close)):
        if close.iloc[i-1] > up2.iloc[i-1]:
            up2.iloc[i] = max(up2_raw.iloc[i], up2.iloc[i-1])
        else:
            up2.iloc[i] = up2_raw.iloc[i]
        if close.iloc[i-1] < dn2.iloc[i-1]:
            dn2.iloc[i] = min(dn2_raw.iloc[i], dn2.iloc[i-1])
        else:
            dn2.iloc[i] = dn2_raw.iloc[i]
    
    trend1 = pd.Series(1, index=close.index)
    for i in range(1, len(close)):
        if trend1.iloc[i-1] == -1 and close.iloc[i] > dn2.iloc[i-1]:
            trend1.iloc[i] = 1
        elif trend1.iloc[i-1] == 1 and close.iloc[i] < up2.iloc[i-1]:
            trend1.iloc[i] = -1
        else:
            trend1.iloc[i] = trend1.iloc[i-1]
    
    # Long condition
    cond = (signal_hull_long.fillna(False) &
            t3_signals_long_cross.fillna(False) &
            (trend1 == 1) &
            (close > ema8) &
            (ema8 > ema20) &
            (ema20 > ema50) &
            (nVolume >= hv))
    
    # Generate entries
    entries = []
    trade_num = 1
    for i in df.index[cond.fillna(False)]:
        entry_ts = int(df.loc[i, 'time'])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        entry_price = float(df.loc[i, 'close'])
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
    
    return entries