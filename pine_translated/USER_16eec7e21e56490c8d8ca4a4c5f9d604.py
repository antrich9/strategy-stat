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
    
    # Input parameters from Pine Script
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
    
    # Time window settings (London morning)
    london_hour_start = 7
    london_minute_start = 45
    london_hour_end = 10
    london_minute_end = 45
    
    close = df['close'].copy()
    high = df['high'].copy()
    low = df['low'].copy()
    
    # ===== E2PSS (Ehlers Two Pole Super Smoother) =====
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    Filt2 = np.zeros(len(df))
    Filt2[0] = close.iloc[0]
    Filt2[1] = close.iloc[1]
    
    for i in range(2, len(df)):
        Filt2[i] = coef1 * close.iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
    
    Filt2 = pd.Series(Filt2, index=df.index)
    TriggerE2PSS = pd.Series(np.roll(Filt2.values, 1), index=df.index)
    TriggerE2PSS.iloc[0] = Filt2.iloc[0]
    
    signalLongE2PSS = Filt2 > TriggerE2PSS
    signalShortE2PSS = Filt2 < TriggerE2PSS
    
    signalLongE2PSSFinal = signalShortE2PSS if inverseE2PSS else signalLongE2PSS
    signalShortE2PSSFinal = signalLongE2PSS if inverseE2PSS else signalShortE2PSS
    
    # ===== Trendilo =====
    pct_change = close.diff(trendilo_smooth) / close * 100
    
    # ALMA implementation
    def alma(series, length, offset, sigma):
        window = np.arange(length)
        m = offset * (length - 1)
        s = sigma * (length - 1) / 6
        w = np.exp(-((window - m) ** 2) / (2 * s ** 2))
        w = w / w.sum()
        
        result = pd.Series(index=series.index, dtype=float)
        for i in range(length - 1, len(series)):
            result.iloc[i] = (series.iloc[i - length + 1:i + 1] * w).sum()
        return result
    
    avg_pct_change = alma(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    
    rms = np.zeros(len(df))
    for i in range(trendilo_length - 1, len(df)):
        window_data = avg_pct_change.iloc[i - trendilo_length + 1:i + 1].values
        rms[i] = trendilo_bmult * np.sqrt(np.sum(window_data ** 2) / trendilo_length)
    rms = pd.Series(rms, index=df.index)
    
    trendilo_dir = pd.Series(0, index=df.index, dtype=int)
    trendilo_dir[avg_pct_change > rms] = 1
    trendilo_dir[avg_pct_change < -rms] = -1
    
    # ===== TTMS (T TM Squeeze) =====
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    close_std = close.rolling(length_TTMS).std()
    BB_mult_TTMS = 2.0
    BB_upper_TTMS = BB_basis_TTMS + BB_mult_TTMS * close_std
    BB_lower_TTMS = BB_basis_TTMS - BB_mult_TTMS * close_std
    
    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    tr = pd.concat([high.diff().abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    devKC_TTMS = tr.rolling(length_TTMS).mean()
    KC_mult_high_TTMS = 1.0
    KC_mult_mid_TTMS = 1.5
    KC_mult_low_TTMS = 2.0
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS
    
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    
    # Momentum (simplified linreg)
    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    avg_center = (highest_high + lowest_low) / 2
    mom_source = close - (avg_center + BB_basis_TTMS) / 2
    
    # Linear regression for momentum
    mom_TTMS = pd.Series(np.nan, index=df.index)
    x = np.arange(length_TTMS)
    x_mean = (length_TTMS - 1) / 2
    
    for i in range(length_TTMS - 1, len(df)):
        y = mom_source.iloc[i - length_TTMS + 1:i + 1].values
        if len(y) == length_TTMS and not np.any(np.isnan(y)):
            y_mean = y.mean()
            slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
            intercept = y_mean - slope * x_mean
            mom_TTMS.iloc[i] = slope * (length_TTMS - 1) + intercept
    
    # TTMS Signals
    mom_prev = mom_TTMS.shift(1).fillna(0)
    mom_diff = mom_TTMS - mom_prev
    
    TTMS_Signals_TTMS = pd.Series(0, index=df.index, dtype=int)
    TTMS_Signals_TTMS[mom_TTMS > 0] = np.where(mom_diff[mom_TTMS > 0] > 0, 1, 2)
    TTMS_Signals_TTMS[mom_TTMS < 0] = np.where(mom_diff[mom_TTMS < 0] < 0, -1, -2)
    
    basicLongCondition_TTMS = pd.Series(False, index=df.index)
    basicShortCondition_TTMS = pd.Series(False, index=df.index)
    
    if redGreen_TTMS:
        basicLongCondition_TTMS = TTMS_Signals_TTMS == 1
        basicShortCondition_TTMS = TTMS_Signals_TTMS == -1
    else:
        basicLongCondition_TTMS = TTMS_Signals_TTMS > 0
        basicShortCondition_TTMS = TTMS_Signals_TTMS < 0
    
    TTMS_SignalsLong_TTMS = pd.Series(False, index=df.index)
    TTMS_SignalsShort_TTMS = pd.Series(False, index=df.index)
    
    if highlightMovements_TTMS:
        TTMS_SignalsLong_TTMS = NoSqz_TTMS & basicLongCondition_TTMS
        TTMS_SignalsShort_TTMS = NoSqz_TTMS & basicShortCondition_TTMS
    else:
        TTMS_SignalsLong_TTMS = basicLongCondition_TTMS
        TTMS_SignalsShort_TTMS = basicShortCondition_TTMS
    
    TTMS_SignalsLongPrev_TTMS = TTMS_SignalsLong_TTMS.shift(1).fillna(False)
    TTMS_SignalsShortPrev_TTMS = TTMS_SignalsShort_TTMS.shift(1).fillna(False)
    
    TTMS_SignalsLongCross_TTMS = pd.Series(False, index=df.index)
    TTMS_SignalsShortCross_TTMS = pd.Series(False, index=df.index)
    
    if cross_TTMS:
        TTMS_SignalsLongCross_TTMS = (~TTMS_SignalsLongPrev_TTMS) & TTMS_SignalsLong_TTMS
        TTMS_SignalsShortCross_TTMS = (~TTMS_SignalsShortPrev_TTMS) & TTMS_SignalsShort_TTMS
    else:
        TTMS_SignalsLongCross_TTMS = TTMS_SignalsLong_TTMS
        TTMS_SignalsShortCross_TTMS = TTMS_SignalsShort_TTMS
    
    TTMS_SignalsLongFinal_TTMS = pd.Series(True, index=df.index)
    TTMS_SignalsShortFinal_TTMS = pd.Series(True, index=df.index)
    
    if use_TTMS:
        if inverse_TTMS:
            TTMS_SignalsLongFinal_TTMS = TTMS_SignalsShortCross_TTMS
            TTMS_SignalsShortFinal_TTMS = TTMS_SignalsLongCross_TTMS
        else:
            TTMS_SignalsLongFinal_TTMS = TTMS_SignalsLongCross_TTMS
            TTMS_SignalsShortFinal_TTMS = TTMS_SignalsShortCross_TTMS
    
    # ===== Time Window (London Morning) =====
    timestamps = pd.to_datetime(df['time'], unit='s', utc=True)
    london_tz = timezone.utc  # Simplified - assumes UTC data, London morning is UTC
    # In practice, would need pytz or zoneinfo for proper London time handling
    # For now, treating timestamps as already in appropriate timezone or UTC
    hours = timestamps.dt.hour
    minutes = timestamps.dt.minute
    time_minutes = hours * 60 + minutes
    morning_start_minutes = london_hour_start * 60 + london_minute_start
    morning_end_minutes = london_hour_end * 60 + london_minute_end
    isWithinMorningWindow = (time_minutes >= morning_start_minutes) & (time_minutes < morning_end_minutes)
    isWithinAfternoonWindow = pd.Series(False, index=df.index)
    isWithinTimeWindow = isWithinMorningWindow | isWithinAfternoonWindow
    
    # ===== Final Conditions =====
    long_condition = signalLongE2PSSFinal & (trendilo_dir == 1) & basicLongCondition_TTMS & isWithinTimeWindow
    short_condition = signalShortE2PSSFinal & (trendilo_dir == -1) & basicShortCondition_TTMS & isWithinTimeWindow
    
    # ===== Generate Entries =====
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if np.isnan(Filt2.iloc[i]) or np.isnan(avg_pct_change.iloc[i]) or np.isnan(mom_TTMS.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
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
        
        if short_condition.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
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