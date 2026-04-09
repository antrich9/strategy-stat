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
    volume = df['volume']
    
    # E2PSS Parameters
    PeriodE2PSS = 15
    useE2PSS = True
    inverseE2PSS = False
    
    # Ehlers 2-Pole Super Smoother
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * np.pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    PriceE2PSS = (high + low) / 2
    
    # Recursive filter calculation
    Filt2 = np.zeros(len(df))
    TriggerE2PSS = np.zeros(len(df))
    
    for i in range(len(df)):
        if i == 0:
            Filt2[i] = PriceE2PSS.iloc[i]
        elif i == 1:
            Filt2[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * Filt2[i-1]
        else:
            Filt2[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
        
        if i == 0:
            TriggerE2PSS[i] = PriceE2PSS.iloc[i]
        else:
            TriggerE2PSS[i] = Filt2[i-1] if not np.isnan(Filt2[i-1]) else PriceE2PSS.iloc[i]
    
    Filt2_series = pd.Series(Filt2, index=df.index)
    TriggerE2PSS_series = pd.Series(TriggerE2PSS, index=df.index)
    
    signalLongE2PSS = Filt2_series > TriggerE2PSS_series
    signalShortE2PSS = Filt2_series < TriggerE2PSS_series
    
    if inverseE2PSS:
        signalLongE2PSSFinal = signalShortE2PSS
        signalShortE2PSSFinal = signalLongE2PSS
    else:
        signalLongE2PSSFinal = signalLongE2PSS
        signalShortE2PSSFinal = signalShortE2PSS
    
    # Trendilo Parameters
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    # Trendilo Calculation
    pct_change = close.pct_change(trendilo_smooth) * 100
    
    def alma(arr, length, offset, sigma):
        """Arnaud Legoux Moving Average"""
        window = np.arange(length)
        m = offset * (length - 1)
        s = sigma * length / 6
        w = np.exp(-(window - m)**2 / (2 * s**2))
        w = w / w.sum()
        
        result = np.full(len(arr), np.nan)
        for i in range(length - 1, len(arr)):
            values = arr.iloc[i - length + 1:i + 1].values
            result[i] = np.sum(w * values)
        return pd.Series(result, index=arr.index)
    
    avg_pct_change = alma(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    
    rms = np.sqrt((avg_pct_change**2).rolling(trendilo_length).mean()) * trendilo_bmult
    
    trendilo_dir = pd.Series(0, index=df.index)
    trendilo_dir[avg_pct_change > rms] = 1
    trendilo_dir[avg_pct_change < -rms] = -1
    
    # TTM Squeeze Parameters
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    redGreen_TTMS = True
    cross_TTMS = True
    inverse_TTMS = False
    highlightMovements_TTMS = True
    use_TTMS = True
    
    # Bollinger Bands
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = close.rolling(length_TTMS).std() * BB_mult_TTMS
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS
    
    # Keltner Channels
    KC_mult_high_TTMS = 1.0
    KC_mult_mid_TTMS = 1.5
    KC_mult_low_TTMS = 2.0
    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    devKC_TTMS = true_range.rolling(length_TTMS).mean() if 'true_range' in dir() else ((high - low).rolling(length_TTMS).mean())
    devKC_TTMS = ((high - low).rolling(length_TTMS).mean() if 'true_range' not in dir() else pd.Series(true_range, index=df.index))
    devKC_TTMS = pd.Series(np.maximum(high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1))), index=df.index).rolling(length_TTMS).mean()
    
    KC_upper_high_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_high_TTMS
    KC_lower_high_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_high_TTMS
    KC_upper_mid_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_mid_TTMS
    KC_lower_mid_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_mid_TTMS
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS
    
    # Squeeze Conditions
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    LowSqz_TTMS = (BB_lower_TTMS >= KC_lower_low_TTMS) | (BB_upper_TTMS <= KC_upper_low_TTMS)
    
    # Momentum Oscillator (linreg)
    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    avg_hl = (highest_high + lowest_low) / 2
    sma_close = close.rolling(length_TTMS).mean()
    momentum_input = close - (avg_hl + sma_close) / 2
    
    def linreg(series, length):
        """Linear regression over length period"""
        result = pd.Series(np.nan, index=series.index)
        for i in range(length - 1, len(series)):
            y = series.iloc[i - length + 1:i + 1].values
            x = np.arange(length)
            x_mean = x.mean()
            y_mean = y.mean()
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            if denominator != 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean
                result.iloc[i] = slope * (length - 1) + intercept
        return result
    
    mom_TTMS = linreg(momentum_input, length_TTMS)
    mom_prev = mom_TTMS.shift(1)
    
    # TTMS Signals
    iff_1_TTMS = pd.Series(1, index=df.index)
    iff_1_TTMS[mom_TTMS <= mom_prev.fillna(mom_TTMS)] = 2
    
    iff_2_TTMS = pd.Series(-1, index=df.index)
    iff_2_TTMS[mom_TTMS >= mom_prev.fillna(mom_TTMS)] = -2
    
    TTMS_Signals_TTMS = pd.Series(0, index=df.index)
    TTMS_Signals_TTMS[mom_TTMS > 0] = iff_1_TTMS[mom_TTMS > 0]
    TTMS_Signals_TTMS[mom_TTMS <= 0] = iff_2_TTMS[mom_TTMS <= 0]
    
    # Basic Conditions
    if redGreen_TTMS:
        basicLongCondition_TTMS = TTMS_Signals_TTMS == 1
        basicShortCondition_TTMS = TTMS_Signals_TTMS == -1
    else:
        basicLongCondition_TTMS = TTMS_Signals_TTMS > 0
        basicShortCondition_TTMS = TTMS_Signals_TTMS < 0
    
    # Heartbeat with squeeze filter
    if highlightMovements_TTMS:
        TTMS_SignalsLong_TTMS = NoSqz_TTMS & basicLongCondition_TTMS
        TTMS_SignalsShort_TTMS = NoSqz_TTMS & basicShortCondition_TTMS
    else:
        TTMS_SignalsLong_TTMS = basicLongCondition_TTMS
        TTMS_SignalsShort_TTMS = basicShortCondition_TTMS
    
    # Cross confirmation
    if cross_TTMS:
        TTMS_SignalsLongCross_TTMS = (~TTMS_SignalsLong_TTMS.shift(1).fillna(False)) & TTMS_SignalsLong_TTMS
        TTMS_SignalsShortCross_TTMS = (~TTMS_SignalsShort_TTMS.shift(1).fillna(False)) & TTMS_SignalsShort_TTMS
    else:
        TTMS_SignalsLongCross_TTMS = TTMS_SignalsLong_TTMS
        TTMS_SignalsShortCross_TTMS = TTMS_SignalsShort_TTMS
    
    # Final signals with inverse option
    if use_TTMS:
        if inverse_TTMS:
            TTMS_SignalsLongFinal_TTMS = TTMS_SignalsShortCross_TTMS
            TTMS_SignalsShortFinal_TTMS = TTMS_SignalsLongCross_TTMS
        else:
            TTMS_SignalsLongFinal_TTMS = TTMS_SignalsLongCross_TTMS
            TTMS_SignalsShortFinal_TTMS = TTMS_SignalsShortCross_TTMS
    else:
        TTMS_SignalsLongFinal_TTMS = pd.Series(True, index=df.index)
        TTMS_SignalsShortFinal_TTMS = pd.Series(True, index=df.index)
    
    # Entry Conditions
    long_condition = signalLongE2PSSFinal & (trendilo_dir == 1) & basicLongCondition_TTMS
    short_condition = signalShortE2PSSFinal & (trendilo_dir == -1) & basicShortCondition_TTMS
    
    # Generate entries
    entries = []
    trade_num = 1
    in_position = False
    position_direction = None
    
    for i in range(len(df)):
        if in_position:
            continue
        
        # Check for valid indicators
        if pd.isna(Filt2_series.iloc[i]) or pd.isna(avg_pct_change.iloc[i]) or pd.isna(mom_TTMS.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            in_position = True
            position_direction = 'long'
        
        elif short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            in_position = True
            position_direction = 'short'
    
    return entries