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
    
    # Extract close and other needed price data
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_price = df['open'].values
    
    # E2PSS Parameters
    useE2PSS = True
    inverseE2PSS = False
    PriceE2PSS = (df['high'] + df['low']) / 2  # hl2
    PeriodE2PSS = 15
    
    # E2PSS Calculation
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    # Initialize Filt2
    Filt2 = np.zeros(len(df))
    TriggerE2PSS = np.zeros(len(df))
    Filt2[0] = PriceE2PSS.iloc[0]
    Filt2[1] = PriceE2PSS.iloc[1]
    
    for i in range(2, len(df)):
        Filt2[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
        TriggerE2PSS[i] = Filt2[i-1]
    
    # Trendilo Parameters
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    # Trendilo ALMA
    def alma(data, length, offset, sigma):
        m = (offset * (length - 1))
        s = sigma * (length - 1) / length
        w = np.exp(-((np.arange(length) - m)**2) / (2 * s * s))
        w_sum = np.sum(w)
        if w_sum == 0:
            return data
        w = w / w_sum
        result = np.convolve(data, w, mode='same')
        return result
    
    # Calculate pct_change
    pct_change = np.zeros(len(df))
    for i in range(trendilo_smooth, len(df)):
        pct_change[i] = (close[i] - close[i - trendilo_smooth]) / close[i - trendilo_smooth] * 100
    
    # Calculate avg_pct_change using ALMA
    avg_pct_change = np.zeros(len(df))
    for i in range(length - 1, len(df)):
        window = pct_change[i - length + 1:i + 1]
        avg_pct_change[i] = np.sum(window * w) / np.sum(w)
    
    # ALMA calculation
    m = (trendilo_offset * (trendilo_length - 1))
    s = trendilo_sigma * (trendilo_length - 1) / trendilo_length
    w = np.exp(-((np.arange(trendilo_length) - m)**2) / (2 * s * s))
    w_norm = w / np.sum(w)
    
    avg_pct_change = np.zeros(len(df))
    for i in range(trendilo_length - 1, len(df)):
        window = pct_change[i - trendilo_length + 1:i + 1]
        avg_pct_change[i] = np.sum(window * w_norm)
    
    # Calculate RMS
    rms_sum = np.zeros(len(df))
    for i in range(trendilo_length - 1, len(df)):
        rms_sum[i] = np.sum(avg_pct_change[i - trendilo_length + 1:i + 1]**2)
    rms = trendilo_bmult * np.sqrt(rms_sum / trendilo_length)
    
    # Trendilo direction
    trendilo_dir = np.zeros(len(df))
    for i in range(len(df)):
        if avg_pct_change[i] > rms[i]:
            trendilo_dir[i] = 1
        elif avg_pct_change[i] < -rms[i]:
            trendilo_dir[i] = -1
        else:
            trendilo_dir[i] = 0
    
    # TTMS Parameters
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    use_TTMS = True
    redGreen_TTMS = True
    cross_TTMS = True
    inverse_TTMS = False
    highlightMovements_TTMS = True
    
    # Bollinger Bands
    BB_basis_TTMS = df['close'].rolling(window=length_TTMS).mean()
    BB_std_TTMS = df['close'].rolling(window=length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + BB_mult_TTMS * BB_std_TTMS
    BB_lower_TTMS = BB_basis_TTMS - BB_mult_TTMS * BB_std_TTMS
    
    # Keltner Channels
    KC_mult_high_TTMS = 1.0
    KC_mult_mid_TTMS = 1.5
    KC_mult_low_TTMS = 2.0
    KC_basis_TTMS = df['close'].rolling(window=length_TTMS).mean()
    devKC_TTMS = df['tr'].rolling(window=length_TTMS).mean()
    KC_upper_high_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_high_TTMS
    KC_lower_high_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_high_TTMS
    KC_upper_mid_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_mid_TTMS
    KC_lower_mid_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_mid_TTMS
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS
    
    # Squeeze conditions
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    LowSqz_TTMS = (BB_lower_TTMS >= KC_lower_low_TTMS) | (BB_upper_TTMS <= KC_upper_low_TTMS)
    MidSqz_TTMS = (BB_lower_TTMS >= KC_lower_mid_TTMS) | (BB_upper_TTMS <= KC_upper_mid_TTMS)
    HighSqz_TTMS = (BB_lower_TTMS >= KC_lower_high_TTMS) | (BB_upper_TTMS <= KC_upper_high_TTMS)
    
    # Momentum
    highest_high = df['high'].rolling(window=length_TTMS).max()
    lowest_low = df['low'].rolling(window=length_TTMS).min()
    sma_close = df['close'].rolling(window=length_TTMS).mean()
    avg_h_l = (highest_high + lowest_low) / 2
    mom_TTMS_raw = close - (avg_h_l + sma_close) / 2
    
    # Linear regression for momentum
    def linreg(x, length):
        result = np.zeros(len(x))
        for i in range(length - 1, len(x)):
            window = x[i - length + 1:i + 1]
            n = len(window)
            if n > 0:
                idx = np.arange(n)
                sum_x = np.sum(idx)
                sum_y = np.sum(window)
                sum_xy = np.sum(idx * window)
                sum_x2 = np.sum(idx**2)
                denom = n * sum_x2 - sum_x**2
                if denom != 0:
                    slope = (n * sum_xy - sum_x * sum_y) / denom
                    intercept = (sum_y - slope * sum_x) / n
                    result[i] = slope * (n - 1) + intercept
        return result
    
    mom_TTMS = linreg(mom_TTMS_raw, length_TTMS)
    
    # TTMS Signals
    iff_1_TTMS_no = np.where(mom_TTMS > np.roll(mom_TTMS, 1), 1, 2)
    iff_2_TTMS_no = np.where(mom_TTMS < np.roll(mom_TTMS, 1), -1, -2)
    
    TTMS_Signals_TTMS = np.where(mom_TTMS > 0, iff_1_TTMS_no, iff_2_TTMS_no)
    
    # Basic conditions
    basicLongCondition_TTMS = np.where(redGreen_TTMS, TTMS_Signals_TTMS == 1, TTMS_Signals_TTMS > 0)
    basicShortCondition_TTMS = np.where(redGreen_TTMS, TTMS_Signals_TTMS == -1, TTMS_Signals_TTMS < 0)
    
    # With highlight
    TTMS_SignalsLong_TTMS = np.where(highlightMovements_TTMS, NoSqz_TTMS & basicLongCondition_TTMS, basicLongCondition_TTMS)
    TTMS_SignalsShort_TTMS = np.where(highlightMovements_TTMS, NoSqz_TTMS & basicShortCondition_TTMS, basicShortCondition_TTMS)
    
    # Cross confirmation
    TTMS_SignalsLong_TTMS_shifted = np.roll(TTMS_SignalsLong_TTMS, 1)
    TTMS_SignalsShort_TTMS_shifted = np.roll(TTMS_SignalsShort_TTMS, 1)
    
    TTMS_SignalsLongCross_TTMS = np.where(cross_TTMS, ~TTMS_SignalsLong_TTMS_shifted & TTMS_SignalsLong_TTMS, TTMS_SignalsLong_TTMS)
    TTMS_SignalsShortCross_TTMS = np.where(cross_TTMS, ~TTMS_SignalsShort_TTMS_shifted & TTMS_SignalsShort_TTMS, TTMS_SignalsShort_TTMS)
    
    # Final signals
    if use_TTMS:
        if inverse_TTMS:
            TTMS_SignalsLongFinal_TTMS = TTMS_SignalsShortCross_TTMS
            TTMS_SignalsShortFinal_TTMS = TTMS_SignalsLongCross_TTMS
        else:
            TTMS_SignalsLongFinal_TTMS = TTMS_SignalsLongCross_TTMS
            TTMS_SignalsShortFinal_TTMS = TTMS_SignalsShortCross_TTMS
    else:
        TTMS_SignalsLongFinal_TTMS = np.ones(len(df))
        TTMS_SignalsShortFinal_TTMS = np.ones(len(df))
    
    # E2PSS signals
    signalLongE2PSS = Filt2 > TriggerE2PSS if useE2PSS else np.ones(len(df), dtype=bool)
    signalShortE2PSS = Filt2 < TriggerE2PSS if useE2PSS else np.ones(len(df), dtype=bool)
    
    if inverseE2PSS:
        signalLongE2PSS_temp = signalLongE2PSS
        signalShortE2PSS_temp = signalShortE2PSS
        signalLongE2PSS = signalShortE2PSS_temp
        signalShortE2PSS = signalLongE2PSS_temp
    
    # Final conditions
    long_condition = signalLongE2PSS & (trendilo_dir == 1) & basicLongCondition_TTMS
    short_condition = signalShortE2PSS & (trendilo_dir == -1) & basicShortCondition_TTMS
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_condition.iloc[i]:
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            }
            entries.append(entry)
            trade_num += 1
        
        if short_condition.iloc[i]:
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            }
            entries.append(entry)
            trade_num += 1
    
    return entries