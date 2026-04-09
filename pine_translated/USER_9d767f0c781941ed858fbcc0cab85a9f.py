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
    
    # Default input values from Pine Script
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
    
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    
    # E2PSS Filter Calculation
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
        if i > 0:
            TriggerE2PSS[i] = Filt2[i-1]
    
    signalLongE2PSS = Filt2 > TriggerE2PSS if useE2PSS else np.ones(len(df), dtype=bool)
    signalShortE2PSS = Filt2 < TriggerE2PSS if useE2PSS else np.ones(len(df), dtype=bool)
    
    signalLongE2PSSFinal = signalShortE2PSS if inverseE2PSS else signalLongE2PSS
    signalShortE2PSSFinal = signalLongE2PSS if inverseE2PSS else signalShortE2PSS
    
    # Trendilo Implementation
    pct_change = close.diff(trendilo_smooth) / close * 100
    avg_pct_change = pct_change.ewm(span=trendilo_length, adjust=False).mean()
    
    rms_list = []
    for i in range(len(df)):
        if i < trendilo_length:
            rms_list.append(np.nan)
        else:
            window = avg_pct_change.iloc[i-trendilo_length+1:i+1]
            rms = trendilo_bmult * np.sqrt((window ** 2).mean())
            rms_list.append(rms)
    rms = pd.Series(rms_list, index=df.index)
    
    trendilo_dir = pd.Series(0, index=df.index)
    trendilo_dir[avg_pct_change > rms] = 1
    trendilo_dir[avg_pct_change < -rms] = -1
    
    # TTM Squeeze Implementation
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = close.rolling(length_TTMS).std() * BB_mult_TTMS
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS
    
    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    devKC_TTMS = df['high'].combine(df['low'], lambda h, l: h - l).rolling(length_TTMS).mean()
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
    
    # Momentum Oscillator (linreg approximation)
    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    avg_price = (highest_high + lowest_low) / 2
    sma_close = close.rolling(length_TTMS).mean()
    momentum_raw = close - avg_price
    mom_TTMS = momentum_raw.rolling(length_TTMS).mean()
    
    mom_prev = mom_TTMS.shift(1)
    TTMS_Signals_TTMS = pd.Series(0, index=df.index)
    TTMS_Signals_TTMS[mom_TTMS > 0] = np.where(mom_TTMS > mom_prev, 1, 2)
    TTMS_Signals_TTMS[mom_TTMS <= 0] = np.where(mom_TTMS < mom_prev, -1, -2)
    
    basicLongCondition_TTMS = (TTMS_Signals_TTMS == 1) if redGreen_TTMS else (TTMS_Signals_TTMS > 0)
    basicShortCondition_TTMS = (TTMS_Signals_TTMS == -1) if redGreen_TTMS else (TTMS_Signals_TTMS < 0)
    
    TTMS_SignalsLong_TTMS = basicLongCondition_TTMS if not highlightMovements_TTMS else (NoSqz_TTMS & basicLongCondition_TTMS)
    TTMS_SignalsShort_TTMS = basicShortCondition_TTMS if not highlightMovements_TTMS else (NoSqz_TTMS & basicShortCondition_TTMS)
    
    TTMS_SignalsLongPrev = TTMS_SignalsLong_TTMS.shift(1).fillna(False)
    TTMS_SignalsShortPrev = TTMS_SignalsShort_TTMS.shift(1).fillna(False)
    
    TTMS_SignalsLongCross_TTMS = TTMS_SignalsLong_TTMS if not cross_TTMS else (~TTMS_SignalsLongPrev & TTMS_SignalsLong_TTMS)
    TTMS_SignalsShortCross_TTMS = TTMS_SignalsShort_TTMS if not cross_TTMS else (~TTMS_SignalsShortPrev & TTMS_SignalsShort_TTMS)
    
    TTMS_SignalsLongFinal_TTMS = TTMS_SignalsShortCross_TTMS if inverse_TTMS else TTMS_SignalsLongCross_TTMS if use_TTMS else pd.Series(True, index=df.index)
    TTMS_SignalsShortFinal_TTMS = TTMS_SignalsLongCross_TTMS if inverse_TTMS else TTMS_SignalsShortCross_TTMS if use_TTMS else pd.Series(True, index=df.index)
    
    # Combined conditions
    long_condition = signalLongE2PSSFinal & (trendilo_dir == 1) & basicLongCondition_TTMS
    short_condition = signalShortE2PSSFinal & (trendilo_dir == -1) & basicShortCondition_TTMS
    
    # Generate entries
    entries = []
    trade_num = 1
    in_position = False
    
    for i in range(len(df)):
        if pd.isna(Filt2[i]) or pd.isna(rms[i]) or pd.isna(mom_TTMS[i]) or pd.isna(BB_basis_TTMS[i]):
            continue
        
        entry_price = close.iloc[i]
        ts = int(df['time'].iloc[i])
        
        if not in_position:
            if long_condition.iloc[i]:
                entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': entry_time,
                    'entry_price_guess': float(entry_price),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(entry_price),
                    'raw_price_b': float(entry_price)
                })
                trade_num += 1
                in_position = True
            elif short_condition.iloc[i]:
                entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': entry_time,
                    'entry_price_guess': float(entry_price),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(entry_price),
                    'raw_price_b': float(entry_price)
                })
                trade_num += 1
                in_position = True
    
    return entries