import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    high = df['high']
    low = df['low']
    close = df['close']
    
    # === FVMA Parameters ===
    adx_length_fvma = 2
    weighting_fvma = 10.0
    ma_length_fvma = 6
    use_fvma = True
    cross_fvma = True
    inverse_fvma = False
    highlight_movements_fvma = True
    
    # === ZeroLagMACD Parameters ===
    fast_length = 12
    slow_length = 26
    signal_length = 9
    
    # === SuperTrend Parameters ===
    atr_period = 10
    atr_multiplier_st = 3.0
    source_st = (high + low) / 2
    
    # Initialize arrays
    n = len(df)
    bulls_fvma = np.zeros(n)
    bears_fvma = np.zeros(n)
    spdi_fvma = np.zeros(n)
    smdi_fvma = np.zeros(n)
    str_fvma = np.zeros(n)
    adx_fvma = np.zeros(n)
    varma_fvma = close.copy().values.astype(float)
    ma_fvma = np.zeros(n)
    
    # FVMA calculation
    for i in range(1, n):
        hi = high.iloc[i]
        hi1 = high.iloc[i-1]
        lo = low.iloc[i]
        lo1 = low.iloc[i-1]
        close1 = close.iloc[i-1]
        
        bulls1 = 0.5 * (abs(hi - hi1) + hi - hi1)
        bears1 = 0.5 * (abs(lo1 - lo) + lo1 - lo)
        
        if bulls1 > bears1:
            bulls_fvma[i] = bulls1
            bears_fvma[i] = 0
        elif bulls1 < bears1:
            bulls_fvma[i] = 0
            bears_fvma[i] = bears1
        else:
            bulls_fvma[i] = 0
            bears_fvma[i] = 0
        
        tr_fvma = max(hi - lo, hi - close1)
        str_fvma[i] = (weighting_fvma * str_fvma[i-1] + tr_fvma) / (weighting_fvma + 1)
        spdi_fvma[i] = (weighting_fvma * spdi_fvma[i-1] + bulls_fvma[i]) / (weighting_fvma + 1)
        smdi_fvma[i] = (weighting_fvma * smdi_fvma[i-1] + bears_fvma[i]) / (weighting_fvma + 1)
        
        pdi = spdi_fvma[i] / str_fvma[i] if str_fvma[i] > 0 else 0
        mdi = smdi_fvma[i] / str_fvma[i] if str_fvma[i] > 0 else 0
        dx = abs(pdi - mdi) / (pdi + mdi) if (pdi + mdi) > 0 else 0
        adx_fvma[i] = (weighting_fvma * adx_fvma[i-1] + dx) / (weighting_fvma + 1)
        
        start_idx = max(0, i - adx_length_fvma + 1)
        adx_window = adx_fvma[start_idx:i+1]
        adxmin = min(1000000.0, np.min(adx_window)) if len(adx_window) > 0 else 1000000.0
        adxmax = max(-1.0, np.max(adx_window)) if len(adx_window) > 0 else -1.0
        diff = adxmax - adxmin
        const_fvma = (adx_fvma[i] - adxmin) / diff if diff > 0 else 0
        
        varma_fvma[i] = ((2 - const_fvma) * varma_fvma[i-1] + const_fvma * close.iloc[i]) / 2
    
    for i in range(ma_length_fvma - 1, n):
        ma_fvma[i] = np.mean(varma_fvma[i - ma_length_fvma + 1:i + 1])
    
    ma_fvma_series = pd.Series(ma_fvma, index=df.index)
    ma_fvma_prev = pd.Series(ma_fvma).shift(1).values
    
    fvma_signals = pd.Series(np.where(ma_fvma > ma_fvma_prev, 1, -1), index=df.index)
    
    basic_long_cond = (fvma_signals > 0) & (close > ma_fvma_series)
    basic_short_cond = (fvma_signals < 0) & (close < ma_fvma_series)
    
    fvma_signals_long = basic_long_cond if (use_fvma and highlight_movements_fvma) else (close > ma_fvma_series if use_fvma else pd.Series([True] * n, index=df.index))
    fvma_signals_short = basic_short_cond if (use_fvma and highlight_movements_fvma) else (close < ma_fvma_series if use_fvma else pd.Series([True] * n, index=df.index))
    
    fvma_signals_long_cross = (~fvma_signals_long.shift(1).fillna(False)) & fvma_signals_long if cross_fvma else fvma_signals_long
    fvma_signals_short_cross = (~fvma_signals_short.shift(1).fillna(False)) & fvma_signals_short if cross_fvma else fvma_signals_short
    
    fvma_signals_long_final = fvma_signals_short_cross if inverse_fvma else fvma_signals_long_cross
    fvma_signals_short_final = fvma_signals_long_cross if inverse_fvma else fvma_signals_short_cross
    
    # ZeroLagMACD (DEMA-based)
    def calc_dema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return 2 * ema1 - ema2
    
    source_macd = close
    ema1_fast = source_macd.ewm(span=fast_length, adjust=False).mean()
    ema2_fast = ema1_fast.ewm(span=fast_length, adjust=False).mean()
    dema_fast = 2 * ema1_fast - ema2_fast
    
    ema1_slow = source_macd.ewm(span=slow_length, adjust=False).mean()
    ema2_slow = ema1_slow.ewm(span=slow_length, adjust=False).mean()
    dema_slow = 2 * ema1_slow - ema2_slow
    
    zero_lag_macd = dema_fast - dema_slow
    
    ema1_sig = zero_lag_macd.ewm(span=signal_length, adjust=False).mean()
    ema2_sig = ema1_sig.ewm(span=signal_length, adjust=False).mean()
    signal_macd = 2 * ema1_sig - ema2_sig
    
    # SuperTrend
    tr_st = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    
    def wilder_smooth(series, length):
        alpha = 1.0 / length
        return series.ewm(alpha=alpha, adjust=False).mean()
    
    atr_st = wilder_smooth(tr_st, atr_period)
    
    up_st = source_st - atr_multiplier_st * atr_st
    up1 = up_st.shift(1).fillna(up_st)
    up_prev_close = close.shift(1)
    up_combined = pd.concat([up_st, up1], axis=1).max(axis=1)
    up = pd.where(up_prev_close > up1, up_combined, up_st)
    up = pd.Series(up, index=df.index)
    
    dn_st = source_st + atr_multiplier_st * atr_st
    dn1 = dn_st.shift(1).fillna(dn_st)
    dn_prev_close = close.shift(1)
    dn_combined = pd.concat([dn_st, dn1], axis=1).min(axis=1)
    dn = pd.where(dn_prev_close < dn1, dn_combined, dn_st)
    dn = pd.Series(dn, index=df.index)
    
    trend = pd.Series(1.0, index=df.index)
    for i in range(1, n):
        prev_trend = trend.iloc[i-1]
        prev_up = up.iloc[i-1] if i > 0 else up.iloc[0]
        prev_dn = dn.iloc[i-1] if i > 0 else dn.iloc[0]
        
        if prev_trend == -1 and close.iloc[i] > prev_dn:
            trend.iloc[i] = 1
        elif prev_trend == 1 and close.iloc[i] < prev_up:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = prev_trend
    
    # Entry condition: trend == -1 AND FVMASignalsShortFinal AND ZeroLagMACD > 0
    entry_condition = (trend == -1) & fvma_signals_short_final & (zero_lag_macd > 0)
    
    entries = []
    trade_num = 1
    
    for i in range(1, n):
        if entry_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = close.iloc[i]
            
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