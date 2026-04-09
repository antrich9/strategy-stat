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
    
    # E2PSS Parameters
    PeriodE2PSS = 15
    useE2PSS = True
    inverseE2PSS = False
    PriceE2PSS = (df['high'] + df['low']) / 2
    
    # E2PSS Calculation
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * np.pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    Filt2 = np.zeros(len(df))
    Filt2[0] = PriceE2PSS.iloc[0]
    Filt2[1] = PriceE2PSS.iloc[1]
    
    for i in range(2, len(df)):
        Filt2[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
    
    TriggerE2PSS = np.roll(Filt2, 1)
    TriggerE2PSS[0] = PriceE2PSS.iloc[0]
    
    signalLongE2PSS = ~useE2PSS | (Filt2 > TriggerE2PSS)
    signalShortE2PSS = ~useE2PSS | (Filt2 < TriggerE2PSS)
    
    signalLongE2PSSFinal = signalShortE2PSS if inverseE2PSS else signalLongE2PSS
    signalShortE2PSSFinal = signalLongE2PSS if inverseE2PSS else signalShortE2PSS
    
    # Trendilo Parameters
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    # Trendilo Calculation
    pct_change = close.diff(trendilo_smooth) / close * 100
    
    def alma_indicator(series, length, offset, sigma):
        m = np.floor(offset * (length - 1))
        s = sigma * (length - 1) / 6
        w = np.exp(-np.arange(length)**2 / (2 * s**2))
        w = w / w.sum()
        result = pd.Series(np.convolve(series, w, mode='same'), index=series.index)
        return result
    
    avg_pct_change = alma_indicator(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    rms = trendilo_bmult * np.sqrt((avg_pct_change**2).rolling(trendilo_length).sum() / trendilo_length)
    trendilo_dir = pd.Series(0, index=close.index)
    trendilo_dir[avg_pct_change > rms] = 1
    trendilo_dir[avg_pct_change < -rms] = -1
    
    # TTMS Parameters
    use_TTMS = True
    redGreen_TTMS = True
    cross_TTMS = True
    inverse_TTMS = False
    highlightMovements_TTMS = True
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    
    # TTMS Bollinger Bands
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    BB_std_TTMS = close.rolling(length_TTMS).std()
    dev_TTMS = BB_mult_TTMS * BB_std_TTMS
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS
    
    # TTMS Keltner Channels
    KC_mult_high_TTMS = 1.0
    KC_mult_mid_TTMS = 1.5
    KC_mult_low_TTMS = 2.0
    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    devKC_TTMS = pd.Series(np.abs(df['high'] - df['low']), index=close.index).rolling(length_TTMS).mean()
    KC_upper_high_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_high_TTMS
    KC_lower_high_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_high_TTMS
    KC_upper_mid_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_mid_TTMS
    KC_lower_mid_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_mid_TTMS
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS
    
    # TTMS Squeeze Conditions
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    LowSqz_TTMS = (BB_lower_TTMS >= KC_lower_low_TTMS) | (BB_upper_TTMS <= KC_upper_low_TTMS)
    MidSqz_TTMS = (BB_lower_TTMS >= KC_lower_mid_TTMS) | (BB_upper_TTMS <= KC_upper_mid_TTMS)
    HighSqz_TTMS = (BB_lower_TTMS >= KC_lower_high_TTMS) | (BB_upper_TTMS <= KC_upper_high_TTMS)
    
    # TTMS Momentum
    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    mom_component = close - ((highest_high + lowest_low) / 2 + BB_basis_TTMS) / 2
    
    def linreg_indicator(series, length):
        result = pd.Series(np.nan, index=series.index)
        x = np.arange(length)
        x_mean = (length - 1) / 2
        sum_x2 = ((x - x_mean) ** 2).sum()
        for i in range(length - 1, len(series)):
            y = series.iloc[i - length + 1:i + 1].values
            y_mean = y.mean()
            sum_xy = ((x - x_mean) * (y - y_mean)).sum()
            result.iloc[i] = sum_xy / sum_x2
        return result
    
    mom_TTMS = linreg_indicator(mom_component, length_TTMS)
    
    iff_1_TTMS_no = pd.Series(1, index=mom_TTMS.index)
    iff_1_TTMS_no[mom_TTMS <= mom_TTMS.shift(1)] = 2
    iff_2_TTMS_no = pd.Series(-1, index=mom_TTMS.index)
    iff_2_TTMS_no[mom_TTMS >= mom_TTMS.shift(1)] = -2
    
    TTMS_Signals_TTMS = pd.Series(0, index=mom_TTMS.index)
    TTMS_Signals_TTMS[mom_TTMS > 0] = iff_1_TTMS_no[mom_TTMS > 0]
    TTMS_Signals_TTMS[mom_TTMS < 0] = iff_2_TTMS_no[mom_TTMS < 0]
    
    if redGreen_TTMS:
        basicLongCondition_TTMS = TTMS_Signals_TTMS == 1
        basicShortCondition_TTMS = TTMS_Signals_TTMS == -1
    else:
        basicLongCondition_TTMS = TTMS_Signals_TTMS > 0
        basicShortCondition_TTMS = TTMS_Signals_TTMS < 0
    
    if highlightMovements_TTMS:
        TTMS_SignalsLong_TTMS = NoSqz_TTMS & basicLongCondition_TTMS
        TTMS_SignalsShort_TTMS = NoSqz_TTMS & basicShortCondition_TTMS
    else:
        TTMS_SignalsLong_TTMS = basicLongCondition_TTMS
        TTMS_SignalsShort_TTMS = basicShortCondition_TTMS
    
    if cross_TTMS:
        TTMS_SignalsLongCross_TTMS = (~TTMS_SignalsLong_TTMS.shift(1).fillna(False).astype(bool)) & TTMS_SignalsLong_TTMS.astype(bool)
        TTMS_SignalsShortCross_TTMS = (~TTMS_SignalsShort_TTMS.shift(1).fillna(False).astype(bool)) & TTMS_SignalsShort_TTMS.astype(bool)
    else:
        TTMS_SignalsLongCross_TTMS = TTMS_SignalsLong_TTMS
        TTMS_SignalsShortCross_TTMS = TTMS_SignalsShort_TTMS
    
    if use_TTMS:
        TTMS_SignalsLongFinal_TTMS = TTMS_SignalsShortCross_TTMS if inverse_TTMS else TTMS_SignalsLongCross_TTMS
        TTMS_SignalsShortFinal_TTMS = TTMS_SignalsLongCross_TTMS if inverse_TTMS else TTMS_SignalsShortCross_TTMS
    else:
        TTMS_SignalsLongFinal_TTMS = pd.Series(True, index=close.index)
        TTMS_SignalsShortFinal_TTMS = pd.Series(True, index=close.index)
    
    # Entry Conditions
    long_condition = signalLongE2PSSFinal & (trendilo_dir == 1) & basicLongCondition_TTMS
    short_condition = signalShortE2PSSFinal & (trendilo_dir == -1) & basicShortCondition_TTMS
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if np.isnan(Filt2[i]) or np.isnan(avg_pct_change.iloc[i]) or np.isnan(mom_TTMS.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
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
        elif short_condition.iloc[i]:
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