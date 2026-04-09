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
    
    # Settings
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
    
    # E2PSS Implementation
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * np.pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    close_arr = close.values
    Filt2 = np.zeros(len(df))
    Filt2[0] = close_arr[0]
    Filt2[1] = close_arr[1]
    
    for i in range(2, len(df)):
        if i < 3:
            Filt2[i] = close_arr[i]
        else:
            Filt2[i] = coef1 * close_arr[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
    
    TriggerE2PSS = np.roll(Filt2, 1)
    TriggerE2PSS[0] = Filt2[0]
    
    Filt2_series = pd.Series(Filt2, index=df.index)
    TriggerE2PSS_series = pd.Series(TriggerE2PSS, index=df.index)
    
    signalLongE2PSS = Filt2_series > TriggerE2PSS
    signalShortE2PSS = Filt2_series < TriggerE2PSS
    
    if inverseE2PSS:
        signalLongE2PSS_final = signalShortE2PSS
        signalShortE2PSS_final = signalLongE2PSS
    else:
        signalLongE2PSS_final = signalLongE2PSS
        signalShortE2PSS_final = signalShortE2PSS
    
    # Trendilo Implementation
    pct_change = close.pct_change(trendilo_smooth) * 100
    avg_pct_change = pct_change.ewm(span=trendilo_length, adjust=False).mean()
    rms = trendilo_bmult * np.sqrt((avg_pct_change ** 2).rolling(trendilo_length).mean())
    trendilo_dir = pd.Series(0, index=df.index)
    trendilo_dir[avg_pct_change > rms] = 1
    trendilo_dir[avg_pct_change < -rms] = -1
    
    # TTM Squeeze Implementation
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = close.rolling(length_TTMS).std() * BB_mult_TTMS
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS
    
    devKC_TTMS = df['high'].rolling(length_TTMS).max() - df['low'].rolling(length_TTMS).min()
    devKC_TTMS = close.rolling(length_TTMS).mean() - close.rolling(length_TTMS).mean() + devKC_TTMS
    devKC_TTMS = (df['high'] - df['low']).rolling(length_TTMS).mean()
    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    KC_upper_high_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_high_TTMS
    KC_lower_high_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_high_TTMS
    KC_upper_mid_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_mid_TTMS
    KC_lower_mid_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_mid_TTMS
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS
    
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    LowSqz_TTMS = (BB_lower_TTMS >= KC_lower_low_TTMS) | (BB_upper_TTMS <= KC_upper_low_TTMS)
    MidSqz_TTMS = (BB_lower_TTMS >= KC_lower_mid_TTMS) | (BB_upper_TTMS <= KC_upper_mid_TTMS)
    HighSqz_TTMS = (BB_lower_TTMS >= KC_lower_high_TTMS) | (BB_upper_TTMS <= KC_upper_high_TTMS)
    
    # TTM Momentum using linreg approximation
    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    avg_center = (highest_high + lowest_low) / 2
    avg_center = (avg_center + close.rolling(length_TTMS).mean()) / 2
    mom_TTMS = (close - avg_center).rolling(length_TTMS).mean() * 2
    
    prev_mom = mom_TTMS.shift(1)
    mom_sign = pd.Series(1, index=df.index)
    mom_sign[mom_TTMS <= 0] = -1
    mom_sign[mom_TTMS > 0] = 2
    mom_sign[(mom_TTMS <= 0) & (mom_TTMS < prev_mom)] = -2
    mom_sign[(mom_TTMS > 0) & (mom_TTMS >= prev_mom)] = 1
    
    TTMS_Signals_TTMS = mom_sign
    
    basicLongCondition_TTMS = (TTMS_Signals_TTMS == 1) if redGreen_TTMS else (TTMS_Signals_TTMS > 0)
    basicShortCondition_TTMS = (TTMS_Signals_TTMS == -1) if redGreen_TTMS else (TTMS_Signals_TTMS < 0)
    
    TTMS_SignalsLong_TTMS = NoSqz_TTMS & basicLongCondition_TTMS if highlightMovements_TTMS else basicLongCondition_TTMS
    TTMS_SignalsShort_TTMS = NoSqz_TTMS & basicShortCondition_TTMS if highlightMovements_TTMS else basicShortCondition_TTMS
    
    prev_long = TTMS_SignalsLong_TTMS.shift(1).fillna(False).astype(bool)
    prev_short = TTMS_SignalsShort_TTMS.shift(1).fillna(False).astype(bool)
    curr_long = TTMS_SignalsLong_TTMS.astype(bool)
    curr_short = TTMS_SignalsShort_TTMS.astype(bool)
    
    TTMS_SignalsLongCross_TTMS = (~prev_long) & curr_long if cross_TTMS else curr_long
    TTMS_SignalsShortCross_TTMS = (~prev_short) & curr_short if cross_TTMS else curr_short
    
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
    
    # Combined Conditions
    long_condition = signalLongE2PSS_final & (trendilo_dir == 1) & basicLongCondition_TTMS
    short_condition = signalShortE2PSS_final & (trendilo_dir == -1) & basicShortCondition_TTMS
    
    entries = []
    trade_num = 1
    in_position = False
    
    for i in range(len(df)):
        if in_position:
            continue
        
        if long_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
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
        elif short_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
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