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
    open_col = df['open']
    time_col = df['time']
    volume = df['volume']
    
    # Inputs
    useE2PSS = True
    inverseE2PSS = False
    PeriodE2PSS = 15
    
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    use_TTMS = True
    redGreen_TTMS = True
    cross_TTMS = True
    inverse_TTMS = False
    highlightMovements_TTMS = True
    
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    
    KC_mult_high_TTMS = 1.0
    KC_mult_mid_TTMS = 1.5
    KC_mult_low_TTMS = 2.0
    
    atrLength = 14
    
    # ─══════════════════════════════════════════════════════
    # E2PSS (Ehlers Two Pole Super Smoother)
    # ─══════════════════════════════════════════════════════
    pi = 2 * np.arcsin(1)
    
    a1 = np.exp(-1.414 * np.pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    PriceE2PSS = (high + low) / 2
    
    Filt2 = np.zeros(len(df))
    TriggerE2PSS = np.zeros(len(df))
    
    for i in range(len(df)):
        if i < 2:
            Filt2[i] = PriceE2PSS.iloc[i]
        else:
            Filt2[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
        TriggerE2PSS[i] = Filt2[i-1] if i > 0 else Filt2[i]
    
    signalLongE2PSS_series = Filt2 > TriggerE2PSS
    signalShortE2PSS_series = Filt2 < TriggerE2PSS
    
    if inverseE2PSS:
        signalLongE2PSSFinal = signalShortE2PSS_series
        signalShortE2PSSFinal = signalLongE2PSS_series
    else:
        signalLongE2PSSFinal = signalLongE2PSS_series
        signalShortE2PSSFinal = signalShortE2PSS_series
    
    # ─══════════════════════════════════════════════════════
    # Trendilo
    # ─══════════════════════════════════════════════════════
    
    # ALMA implementation
    def alma(series, window, offset=0.85, sigma=6):
        w = np.arange(window)
        m = offset * (window - 1)
        s = window / sigma
        w = np.exp(-((w - m) ** 2) / (2 * s * s))
        w = w / w.sum()
        
        result = np.zeros(len(series))
        for i in range(window - 1, len(series)):
            result[i] = np.sum(w * series.iloc[i - window + 1:i + 1].values)
        return pd.Series(result, index=series.index)
    
    pct_change = (close.diff(trendilo_smooth) / close) * 100
    avg_pct_change = alma(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    
    def rolling_sum(series, length):
        return series.rolling(length).sum()
    
    rms = trendilo_bmult * np.sqrt(rolling_sum(avg_pct_change * avg_pct_change, trendilo_length) / trendilo_length)
    trendilo_dir = np.where(avg_pct_change > rms, 1, np.where(avg_pct_change < -rms, -1, 0))
    trendilo_dir = pd.Series(trendilo_dir, index=close.index)
    
    # ─══════════════════════════════════════════════════════
    # TTM Squeeze (TTMS)
    # ─══════════════════════════════════════════════════════
    
    # Bollinger Bands
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = BB_mult_TTMS * close.rolling(length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS
    
    # Keltner Channels
    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    devKC_TTMS = low.rolling(length_TTMS).mean()  # Using low as proxy for tr in this context
    
    KC_upper_high_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_high_TTMS
    KC_lower_high_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_high_TTMS
    KC_upper_mid_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_mid_TTMS
    KC_lower_mid_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_mid_TTMS
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS
    
    # Squeeze Conditions
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    LowSqz_TTMS = (BB_lower_TTMS >= KC_lower_low_TTMS) | (BB_upper_TTMS <= KC_upper_low_TTMS)
    MidSqz_TTMS = (BB_lower_TTMS >= KC_lower_mid_TTMS) | (BB_upper_TTMS <= KC_upper_mid_TTMS)
    HighSqz_TTMS = (BB_lower_TTMS >= KC_lower_high_TTMS) | (BB_upper_TTMS <= KC_upper_high_TTMS)
    
    # Momentum (linreg approach)
    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    avg_of_avg = (highest_high + lowest_low) / 2
    mom_TTMS_raw = close - avg_of_avg
    
    # Simplified momentum
    mom_TTMS = mom_TTMS_raw.rolling(length_TTMS).mean()
    
    # Momentum direction
    mom_prev = mom_TTMS.shift(1)
    iff_1_TTMS_no = np.where(mom_TTMS > mom_prev, 1, 2)
    iff_2_TTMS_no = np.where(mom_TTMS < mom_prev, -1, -2)
    
    TTMS_Signals_TTMS = np.where(mom_TTMS > 0, iff_1_TTMS_no, iff_2_TTMS_no)
    TTMS_Signals_TTMS = pd.Series(TTMS_Signals_TTMS, index=close.index)
    
    # Basic conditions
    if redGreen_TTMS:
        basicLongCondition_TTMS = TTMS_Signals_TTMS == 1
        basicShortCondition_TTMS = TTMS_Signals_TTMS == -1
    else:
        basicLongCondition_TTMS = TTMS_Signals_TTMS > 0
        basicShortCondition_TTMS = TTMS_Signals_TTMS < 0
    
    # Heartbeat signals
    if highlightMovements_TTMS:
        TTMS_SignalsLong_TTMS = NoSqz_TTMS & basicLongCondition_TTMS
        TTMS_SignalsShort_TTMS = NoSqz_TTMS & basicShortCondition_TTMS
    else:
        TTMS_SignalsLong_TTMS = basicLongCondition_TTMS
        TTMS_SignalsShort_TTMS = basicShortCondition_TTMS
    
    # Cross confirmation
    if cross_TTMS:
        TTMS_SignalsLongCross_TTMS = (~TTMS_SignalsLong_TTMS.shift(1).fillna(False).astype(bool)) & TTMS_SignalsLong_TTMS.astype(bool)
        TTMS_SignalsShortCross_TTMS = (~TTMS_SignalsShort_TTMS.shift(1).fillna(False).astype(bool)) & TTMS_SignalsShort_TTMS.astype(bool)
    else:
        TTMS_SignalsLongCross_TTMS = TTMS_SignalsLong_TTMS
        TTMS_SignalsShortCross_TTMS = TTMS_SignalsShort_TTMS
    
    # Final signals
    if use_TTMS:
        if inverse_TTMS:
            TTMS_SignalsLongFinal_TTMS = TTMS_SignalsShortCross_TTMS
            TTMS_SignalsShortFinal_TTMS = TTMS_SignalsLongCross_TTMS
        else:
            TTMS_SignalsLongFinal_TTMS = TTMS_SignalsLongCross_TTMS
            TTMS_SignalsShortFinal_TTMS = TTMS_SignalsShortCross_TTMS
    else:
        TTMS_SignalsLongFinal_TTMS = pd.Series(True, index=close.index)
        TTMS_SignalsShortFinal_TTMS = pd.Series(True, index=close.index)
    
    # ─══════════════════════════════════════════════════════
    # Entry Conditions
    # ─══════════════════════════════════════════════════════
    
    long_condition = signalLongE2PSSFinal & (trendilo_dir == 1) & basicLongCondition_TTMS
    short_condition = signalShortE2PSSFinal & (trendilo_dir == -1) & basicShortCondition_TTMS
    
    # ─══════════════════════════════════════════════════════
    # Generate Entries
    # ─══════════════════════════════════════════════════════
    
    entries = []
    trade_num = 1
    
    # Convert boolean series to numpy arrays for iteration
    long_cond_arr = long_condition.values
    short_cond_arr = short_condition.values
    
    for i in range(len(df)):
        if np.isnan(Filt2[i]) or np.isnan(avg_pct_change.iloc[i]) if i < len(avg_pct_change) else True:
            continue
        if np.isnan(mom_TTMS.iloc[i]) if i < len(mom_TTMS) else True:
            continue
            
        if long_cond_arr[i]:
            ts = int(time_col.iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
            
        if short_cond_arr[i]:
            ts = int(time_col.iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
    
    return entries