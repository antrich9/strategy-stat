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
    useE2PSS = True
    inverseE2PSS = False
    PeriodE2PSS = 15
    
    # E2PSS Implementation
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * np.pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    # Calculate Filt2
    Filt2 = np.zeros(len(df))
    Filt2[0] = close.iloc[0]
    Filt2[1] = close.iloc[1]
    
    for i in range(2, len(df)):
        Filt2[i] = coef1 * close.iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
    
    TriggerE2PSS = np.roll(Filt2, 1)
    TriggerE2PSS[0] = Filt2[0]
    
    # E2PSS Signals
    signalLongE2PSS = Filt2 > TriggerE2PSS if useE2PSS else np.ones(len(df), dtype=bool)
    signalShortE2PSS = Filt2 < TriggerE2PSS if useE2PSS else np.ones(len(df), dtype=bool)
    
    if inverseE2PSS:
        signalLongE2PSS_final = signalShortE2PSS
        signalShortE2PSS_final = signalLongE2PSS
    else:
        signalLongE2PSS_final = signalLongE2PSS
        signalShortE2PSS_final = signalShortE2PSS
    
    # Trendilo Parameters
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    # Trendilo Implementation
    pct_change = close.pct_change(trendilo_smooth) * 100
    avg_pct_change = pct_change.ewm(span=trendilo_length, adjust=False).mean()
    rms = trendilo_bmult * np.sqrt((avg_pct_change ** 2).rolling(trendilo_length).mean())
    trendilo_dir = np.where(avg_pct_change > rms, 1, np.where(avg_pct_change < -rms, -1, 0))
    
    # TTM Squeeze Parameters
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    
    # Bollinger Bands
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = BB_mult_TTMS * close.rolling(length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS
    
    # Keltner Channels
    KC_mult_high_TTMS = 1.0
    KC_mult_mid_TTMS = 1.5
    KC_mult_low_TTMS = 2.0
    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    devKC_TTMS = tr.rolling(length_TTMS).mean()
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
    
    # TTM Momentum
    highesthigh = high.rolling(length_TTMS).max()
    lowestlow = low.rolling(length_TTMS).min()
    mom_TTMS = (close - ((high.rolling(length_TTMS).mean() + lowestlow) / 2 + close.rolling(length_TTMS).mean()) / 2).rolling(length_TTMS).mean()
    
    # Momentum Histogram Color
    mom_color_TTMS = np.where(mom_TTMS > 0, np.where(mom_TTMS > mom_TTMS.shift(1), 1, 2), np.where(mom_TTMS < mom_TTMS.shift(1), -1, -2))
    
    # Heartbeat Logic
    TTMS_Signals_TTMS = np.where(mom_TTMS > 0, 1, -1)
    
    # TTM Signals
    redGreen_TTMS = True
    highlightMovements_TTMS = True
    use_TTMS = True
    inverse_TTMS = False
    cross_TTMS = True
    
    basicLongCondition_TTMS = (TTMS_Signals_TTMS == 1) if redGreen_TTMS else (TTMS_Signals_TTMS > 0)
    basicShortCondition_TTMS = (TTMS_Signals_TTMS == -1) if redGreen_TTMS else (TTMS_Signals_TTMS < 0)
    
    TTMS_SignalsLong_TTMS = basicLongCondition_TTMS if not highlightMovements_TTMS else (NoSqz_TTMS & basicLongCondition_TTMS)
    TTMS_SignalsShort_TTMS = basicShortCondition_TTMS if not highlightMovements_TTMS else (NoSqz_TTMS & basicShortCondition_TTMS)
    
    # TTM Cross Confirmation
    if cross_TTMS:
        TTMS_SignalsLongCross_TTMS = ~TTMS_SignalsLong_TTMS.shift(1).fillna(False) & TTMS_SignalsLong_TTMS
        TTMS_SignalsShortCross_TTMS = ~TTMS_SignalsShort_TTMS.shift(1).fillna(False) & TTMS_SignalsShort_TTMS
    else:
        TTMS_SignalsLongCross_TTMS = TTMS_SignalsLong_TTMS
        TTMS_SignalsShortCross_TTMS = TTMS_SignalsShort_TTMS
    
    # Final Signals
    if use_TTMS:
        if inverse_TTMS:
            TTMS_SignalsLongFinal_TTMS = TTMS_SignalsShortCross_TTMS
            TTMS_SignalsShortFinal_TTMS = TTMS_SignalsLongCross_TTMS
        else:
            TTMS_SignalsLongFinal_TTMS = TTMS_SignalsLongCross_TTMS
            TTMS_SignalsShortFinal_TTMS = TTMS_SignalsShortCross_TTMS
    else:
        TTMS_SignalsLongFinal_TTMS = np.ones(len(df), dtype=bool)
        TTMS_SignalsShortFinal_TTMS = np.ones(len(df), dtype=bool)
    
    # Entry Conditions
    long_condition = signalLongE2PSS_final & (trendilo_dir == 1) & basicLongCondition_TTMS & TTMS_SignalsLongFinal_TTMS
    short_condition = signalShortE2PSS_final & (trendilo_dir == -1) & basicShortCondition_TTMS & TTMS_SignalsShortFinal_TTMS
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
    
    return entries