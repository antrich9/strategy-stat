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
    
    # Input parameters
    useE2PSS = True
    inverseE2PSS = False
    PeriodE2PSS = 15
    PriceE2PSS = (high + low) / 2
    
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
    
    length_ci = 14
    atrLength = 14
    
    # --- E2PSS Filter ---
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    Filt2 = pd.Series(0.0, index=df.index)
    TriggerE2PSS = pd.Series(0.0, index=df.index)
    
    for i in range(3, len(df)):
        Filt2.iloc[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * Filt2.iloc[i-1] + coef3 * Filt2.iloc[i-2]
    for i in range(2, len(df)):
        TriggerE2PSS.iloc[i] = Filt2.iloc[i-1]
    
    signalLongE2PSS = Filt2 > TriggerE2PSS if useE2PSS else pd.Series(True, index=df.index)
    signalShortE2PSS = Filt2 < TriggerE2PSS if useE2PSS else pd.Series(True, index=df.index)
    
    signalLongE2PSSFinal = signalShortE2PSS if inverseE2PSS else signalLongE2PSS
    signalShortE2PSSFinal = signalLongE2PSS if inverseE2PSS else signalShortE2PSS
    
    # --- Trendilo ---
    pct_change = close.diff(trendilo_smooth) / close * 100
    
    def alma(data, length, offset, sigma):
        m = np.floor(offset * (length - 1))
        s = length / sigma
        weights = np.exp(-((np.arange(length) - m) ** 2) / (2 * s * s))
        weights = weights / weights.sum()
        result = pd.Series(0.0, index=data.index)
        for i in range(length - 1, len(data)):
            result.iloc[i] = np.sum(data.iloc[i-length+1:i+1].values * weights)
        return result
    
    avg_pct_change = alma(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    
    rms = pd.Series(0.0, index=df.index)
    for i in range(trendilo_length - 1, len(df)):
        rms.iloc[i] = trendilo_bmult * np.sqrt(np.mean(avg_pct_change.iloc[i-trendilo_length+1:i+1].values ** 2))
    
    trendilo_dir = pd.Series(0, index=df.index)
    trendilo_dir = np.where(avg_pct_change > rms, 1, np.where(avg_pct_change < -rms, -1, 0))
    trendilo_dir = pd.Series(trendilo_dir, index=df.index)
    
    # --- TTMS ---
    BB_basis = close.rolling(length_TTMS).mean()
    dev_BB = BB_mult_TTMS * close.rolling(length_TTMS).std()
    BB_upper = BB_basis + dev_BB
    BB_lower = BB_basis - dev_BB
    
    KC_mult_high = 1.0
    KC_mult_mid = 1.5
    KC_mult_low = 2.0
    KC_basis = close.rolling(length_TTMS).mean()
    
    tr_kc = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    
    def wilder_smooth(series, period):
        result = pd.Series(0.0, index=series.index)
        alpha = 1 / period
        result.iloc[period-1] = series.iloc[:period].mean()
        for i in range(period, len(series)):
            result.iloc[i] = series.iloc[i] * alpha + result.iloc[i-1] * (1 - alpha)
        return result
    
    devKC = wilder_smooth(tr_kc, length_TTMS)
    
    KC_upper_high = KC_basis + devKC * KC_mult_high
    KC_lower_high = KC_basis - devKC * KC_mult_high
    KC_upper_mid = KC_basis + devKC * KC_mult_mid
    KC_lower_mid = KC_basis - devKC * KC_mult_mid
    KC_upper_low = KC_basis + devKC * KC_mult_low
    KC_lower_low = KC_basis - devKC * KC_mult_low
    
    NoSqz_TTMS = (BB_lower < KC_lower_low) | (BB_upper > KC_upper_low)
    LowSqz_TTMS = (BB_lower >= KC_lower_low) | (BB_upper <= KC_upper_low)
    MidSqz_TTMS = (BB_lower >= KC_lower_mid) | (BB_upper <= KC_upper_mid)
    HighSqz_TTMS = (BB_lower >= KC_lower_high) | (BB_upper <= KC_upper_high)
    
    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    avg_of_avgs = (highest_high + lowest_low) / 2
    sma_close = close.rolling(length_TTMS).mean()
    
    mom_TTMS = pd.Series(0.0, index=df.index)
    for i in range(length_TTMS - 1, len(df)):
        window_close = close.iloc[i-length_TTMS+1:i+1].values
        window_avg = avg_of_avgs.iloc[i-length_TTMS+1:i+1].values
        deviation = window_close - window_avg
        x = np.arange(length_TTMS)
        x_mean = x.mean()
        slope = np.sum((x - x_mean) * deviation) / np.sum((x - x_mean) ** 2)
        intercept = deviation.mean() - slope * x_mean
        mom_TTMS.iloc[i] = slope * (length_TTMS - 1) + intercept
    
    iff_1_TTMS_no = pd.Series(np.where(mom_TTMS > mom_TTMS.shift(1), 1, 2), index=df.index)
    iff_2_TTMS_no = pd.Series(np.where(mom_TTMS < mom_TTMS.shift(1), -1, -2), index=df.index)
    
    TTMS_Signals_TTMS = pd.Series(np.where(mom_TTMS > 0, iff_1_TTMS_no, iff_2_TTMS_no), index=df.index)
    
    basicLongCondition_TTMS = (TTMS_Signals_TTMS == 1) if redGreen_TTMS else (TTMS_Signals_TTMS > 0)
    basicShortCondition_TTMS = (TTMS_Signals_TTMS == -1) if redGreen_TTMS else (TTMS_Signals_TTMS < 0)
    
    TTMS_SignalsLong_TTMS = (NoSqz_TTMS & basicLongCondition_TTMS) if highlightMovements_TTMS else basicLongCondition_TTMS
    TTMS_SignalsShort_TTMS = (NoSqz_TTMS & basicShortCondition_TTMS) if highlightMovements_TTMS else basicShortCondition_TTMS
    
    TTMS_SignalsLongCross_TTMS = (~TTMS_SignalsLong_TTMS.shift(1).fillna(False) & TTMS_SignalsLong_TTMS) if cross_TTMS else TTMS_SignalsLong_TTMS
    TTMS_SignalsShortCross_TTMS = (~TTMS_SignalsShort_TTMS.shift(1).fillna(False) & TTMS_SignalsShort_TTMS) if cross_TTMS else TTMS_SignalsShort_TTMS
    
    TTMS_SignalsLongFinal_TTMS = (TTMS_SignalsShortCross_TTMS if inverse_TTMS else TTMS_SignalsLongCross_TTMS) if use_TTMS else pd.Series(True, index=df.index)
    TTMS_SignalsShortFinal_TTMS = (TTMS_SignalsLongCross_TTMS if inverse_TTMS else TTMS_SignalsShortCross_TTMS) if use_TTMS else pd.Series(True, index=df.index)
    
    # --- CI (Choppiness Index) ---
    tr_ci = high - low
    sum_atr_ci = tr_ci.rolling(length_ci).sum()
    highest_ci = high.rolling(length_ci).max()
    lowest_ci = low.rolling(length_ci).min()
    ci = 100 * np.log10(sum_atr_ci / (highest_ci - lowest_ci)) / np.log10(length_ci)
    
    # --- Volume condition (enoughVol) ---
    atr1 = wilder_smooth(tr_kc, atrLength)
    atrMed = atr1.rolling(50).mean()
    enoughVol = atr1 > atrMed * 1.0
    
    # --- Final Conditions ---
    long_condition = (ci < 50) & signalLongE2PSSFinal & (trendilo_dir == 1) & basicLongCondition_TTMS & enoughVol
    short_condition = (ci < 50) & signalShortE2PSSFinal & (trendilo_dir == -1) & basicShortCondition_TTMS & enoughVol
    
    # --- Generate Entries ---
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(ci.iloc[i]) or pd.isna(signalLongE2PSSFinal.iloc[i]) or pd.isna(trendilo_dir.iloc[i]) or pd.isna(basicLongCondition_TTMS.iloc[i]) or pd.isna(enoughVol.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = close.iloc[i]
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
        elif short_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = close.iloc[i]
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
    
    return entries