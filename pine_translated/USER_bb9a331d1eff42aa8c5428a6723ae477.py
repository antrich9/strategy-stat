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
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    # JMA Parameters
    lengthjmaJMA = 7
    phasejmaJMA = 50
    powerjmaJMA = 2
    usejmaJMA = True
    usecolorjmaJMA = True
    inverseJMA = True
    highlightMovementsjmaJMA = True
    
    # Trendilo Parameters
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    # TTM Squeeze Parameters
    use_TTMS = True
    redGreen_TTMS = True
    cross_TTMS = True
    inverse_TTMS = False
    highlightMovements_TTMS = True
    length_TTMS = 20
    
    BB_mult_TTMS = 2.0
    
    # ─────────────────────────────────────────
    # JMA Calculation (Iterative)
    # ─────────────────────────────────────────
    phasejmaJMARatiojmaJMA = 0.5 if phasejmaJMA < -100 else (2.5 if phasejmaJMA > 100 else phasejmaJMA / 100 + 1.5)
    betajmaJMA = 0.45 * (lengthjmaJMA - 1) / (0.45 * (lengthjmaJMA - 1) + 2)
    alphajmaJMA = betajmaJMA ** powerjmaJMA
    
    jmaJMA = np.zeros(len(df))
    e0JMA = np.zeros(len(df))
    e1JMA = np.zeros(len(df))
    e2JMA = np.zeros(len(df))
    
    for i in range(1, len(df)):
        src = close.iloc[i]
        e0JMA[i] = (1 - alphajmaJMA) * src + alphajmaJMA * e0JMA[i-1]
        e1JMA[i] = (src - e0JMA[i]) * (1 - betajmaJMA) + betajmaJMA * e1JMA[i-1]
        e2JMA[i] = (e0JMA[i] + phasejmaJMARatiojmaJMA * e1JMA[i] - jmaJMA[i-1]) * ((1 - alphajmaJMA) ** 2) + (alphajmaJMA ** 2) * e2JMA[i-1]
        jmaJMA[i] = e2JMA[i] + jmaJMA[i-1]
    
    jmaJMA = pd.Series(jmaJMA, index=df.index)
    jmaJMA_prev = jmaJMA.shift(1)
    
    # JMA Signal Conditions
    signalmaJMALong = True if not usejmaJMA else ((jmaJMA > jmaJMA_prev) & (close > jmaJMA) if usecolorjmaJMA else (close > jmaJMA))
    signalmaJMAShort = True if not usejmaJMA else ((jmaJMA < jmaJMA_prev) & (close < jmaJMA) if usecolorjmaJMA else (close < jmaJMA))
    
    finalLongSignalJMA = signalmaJMAShort if inverseJMA else signalmaJMALong
    finalShortSignalJMA = signalmaJMALong if inverseJMA else signalmaJMAShort
    
    # ─────────────────────────────────────────
    # Trendilo Calculation
    # ─────────────────────────────────────────
    pct_change = close.pct_change(periods=trendilo_smooth) * 100
    
    def alma(arr, length, offset, sigma):
        window = np.arange(length)
        w = np.exp(-((window - offset * (length - 1)) ** 2) / (2 * sigma ** 2))
        w = w / w.sum()
        return pd.Series(np.convolve(arr, w, mode='same'), index=arr.index)
    
    avg_pct_change = alma(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    
    def rolling_rms(x, window):
        return np.sqrt(np.convolve(x**2, np.ones(window)/window, mode='same'))
    
    rms = trendilo_bmult * rolling_rms(avg_pct_change.values, trendilo_length)
    trendilo_dir = np.where(avg_pct_change.values > rms, 1, np.where(avg_pct_change.values < -rms, -1, 0))
    trendilo_dir = pd.Series(trendilo_dir, index=df.index)
    
    # ─────────────────────────────────────────
    # TTM Squeeze Calculation
    # ─────────────────────────────────────────
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = BB_mult_TTMS * close.rolling(length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS
    
    devKC_TTMS = high.rolling(length_TTMS).mean() - low.rolling(length_TTMS).mean()
    KC_lower_low_TTMS = BB_basis_TTMS - devKC_TTMS * 2.0
    KC_upper_low_TTMS = BB_basis_TTMS + devKC_TTMS * 2.0
    KC_lower_mid_TTMS = BB_basis_TTMS - devKC_TTMS * 1.5
    KC_upper_mid_TTMS = BB_basis_TTMS + devKC_TTMS * 1.5
    KC_lower_high_TTMS = BB_basis_TTMS - devKC_TTMS * 1.0
    KC_upper_high_TTMS = BB_basis_TTMS + devKC_TTMS * 1.0
    
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    NoSqz_TTMS = NoSqz_TTMS.fillna(False)
    
    # Momentum Oscillator
    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    avg_high_low = (highest_high + lowest_low) / 2
    avg_close_sma = close.rolling(length_TTMS).mean()
    momentum_raw = close - (avg_high_low + avg_close_sma) / 2
    
    def linreg(series, length):
        x = np.arange(length)
        x_mean = (length - 1) / 2
        result = np.zeros(len(series))
        for i in range(length - 1, len(series)):
            y = series.iloc[i-length+1:i+1].values
            y_mean = np.mean(y)
            cov = np.sum((x - x_mean) * (y - y_mean))
            var = np.sum((x - x_mean) ** 2)
            slope = cov / var if var != 0 else 0
            intercept = y_mean - slope * x_mean
            result[i] = slope * (length - 1) + intercept
        return pd.Series(result, index=series.index)
    
    mom_TTMS = linreg(momentum_raw, length_TTMS)
    mom_TTMS_filled = mom_TTMS.fillna(0)
    mom_TTMS_prev = mom_TTMS_filled.shift(1).fillna(0)
    
    TTMS_Signals_TTMS = np.where(mom_TTMS_filled > 0,
                                  np.where(mom_TTMS_filled > mom_TTMS_prev, 1, 2),
                                  np.where(mom_TTMS_filled < mom_TTMS_prev, -1, -2))
    TTMS_Signals_TTMS = pd.Series(TTMS_Signals_TTMS, index=df.index)
    
    basicLongCondition_TTMS = (TTMS_Signals_TTMS == 1) if redGreen_TTMS else (TTMS_Signals_TTMS > 0)
    basicShortCondition_TTMS = (TTMS_Signals_TTMS == -1) if redGreen_TTMS else (TTMS_Signals_TTMS < 0)
    
    NoSqz_arr = NoSqz_TTMS.values
    TTMS_SignalsLong_TTMS = np.where(highlightMovements_TTMS,
                                      NoSqz_arr & basicLongCondition_TTMS.values,
                                      basicLongCondition_TTMS.values)
    TTMS_SignalsShort_TTMS = np.where(highlightMovements_TTMS,
                                       NoSqz_arr & basicShortCondition_TTMS.values,
                                       basicShortCondition_TTMS.values)
    
    TTMS_SignalsLong_TTMS = pd.Series(TTMS_SignalsLong_TTMS, index=df.index)
    TTMS_SignalsShort_TTMS = pd.Series(TTMS_SignalsShort_TTMS, index=df.index)
    
    TTMS_SignalsLong_TTMS_prev = TTMS_SignalsLong_TTMS.shift(1).fillna(False).astype(bool)
    TTMS_SignalsShort_TTMS_prev = TTMS_SignalsShort_TTMS.shift(1).fillna(False).astype(bool)
    TTMS_SignalsLong_TTMS_curr = TTMS_SignalsLong_TTMS.astype(bool)
    TTMS_SignalsShort_TTMS_curr = TTMS_SignalsShort_TTMS.astype(bool)
    
    TTMS_SignalsLongCross_TTMS = (~TTMS_SignalsLong_TTMS_prev & TTMS_SignalsLong_TTMS_curr) if cross_TTMS else TTMS_SignalsLong_TTMS_curr
    TTMS_SignalsShortCross_TTMS = (~TTMS_SignalsShort_TTMS_prev & TTMS_SignalsShort_TTMS_curr) if cross_TTMS else TTMS_SignalsShort_TTMS_curr
    
    TTMS_SignalsLongFinal_TTMS = TTMS_SignalsShortCross_TTMS if (use_TTMS and inverse_TTMS) else (TTMS_SignalsLongCross_TTMS if use_TTMS else pd.Series(True, index=df.index))
    TTMS_SignalsShortFinal_TTMS = TTMS_SignalsLongCross_TTMS if (use_TTMS and inverse_TTMS) else (TTMS_SignalsShortCross_TTMS if use_TTMS else pd.Series(True, index=df.index))
    
    # ─────────────────────────────────────────
    # Entry Conditions
    # ─────────────────────────────────────────
    long_condition = signalmaJMALong & (trendilo_dir == 1) & basicLongCondition_TTMS
    short_condition = signalmaJMAShort & (trendilo_dir == -1) & basicShortCondition_TTMS
    
    long_condition = long_condition.fillna(False)
    short_condition = short_condition.fillna(False)
    
    # ─────────────────────────────────────────
    # Generate Entries
    # ─────────────────────────────────────────
    entries = []
    trade_num = 1
    in_position = False
    
    for i in range(len(df)):
        if in_position:
            continue
        
        entry_price = close.iloc[i]
        ts = int(df['time'].iloc[i])
        
        if long_condition.iloc[i]:
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
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
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time_str,
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