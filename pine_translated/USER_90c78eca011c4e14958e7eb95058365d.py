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
    # Parameters from Pine Script inputs
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
    
    trendPeriod = 20
    maType1 = 'JMA'
    maPeriod = 8
    
    # Placeholder for CRYPTOCAP:TOTAL - use close price if not available
    symbolData = df['close'].copy()
    
    # Initialize result list
    entries = []
    trade_num = 1
    
    # Prepare price series
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    
    n = len(df)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # E2PSS Implementation
    # ═══════════════════════════════════════════════════════════════════════════════
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * np.pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    PriceE2PSS = (high + low) / 2
    
    Filt2 = np.zeros(n)
    TriggerE2PSS = np.zeros(n)
    
    for i in range(n):
        if i < 2:
            Filt2[i] = PriceE2PSS.iloc[i]
        else:
            Filt2[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
        if i > 0:
            TriggerE2PSS[i] = Filt2[i-1]
    
    signalLongE2PSS = ~useE2PSS | (Filt2 > TriggerE2PSS)
    signalShortE2PSS = ~useE2PSS | (Filt2 < TriggerE2PSS)
    
    signalLongE2PSSFinal = ~inverseE2PSS & signalLongE2PSS | inverseE2PSS & signalShortE2PSS
    signalShortE2PSSFinal = ~inverseE2PSS & signalShortE2PSS | inverseE2PSS & signalLongE2PSS
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # Trendilo Implementation
    # ═══════════════════════════════════════════════════════════════════════════════
    pct_change = close.pct_change(trendilo_smooth) * 100
    avg_pct_change = pct_change.ewm(span=trendilo_length, adjust=False).mean()
    rms = trendilo_bmult * np.sqrt((avg_pct_change ** 2).rolling(trendilo_length).mean())
    trendilo_dir = np.where(avg_pct_change > rms, 1, np.where(avg_pct_change < -rms, -1, 0))
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # TTMS Implementation
    # ═══════════════════════════════════════════════════════════════════════════════
    # Bollinger Bands
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = BB_mult_TTMS * close.rolling(length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS
    
    # Keltner Channels
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    devKC_TTMS = tr.rolling(length_TTMS).mean()
    KC_basis_TTMS = BB_basis_TTMS
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
    
    # Momentum Oscillator
    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    mid_price = (highest_high + lowest_low) / 2
    mom_TTMS_raw = close - mid_price
    mom_TTMS = mom_TTMS_raw.rolling(length_TTMS).mean()
    
    # Momentum signals
    mom_increasing = mom_TTMS > mom_TTMS.shift(1)
    iff_1_TTMS_no = np.where(mom_increasing, 1, 2)
    iff_2_TTMS_no = np.where(mom_increasing, -1, -2)
    
    TTMS_Signals_TTMS = np.where(mom_TTMS > 0, iff_1_TTMS_no, iff_2_TTMS_no)
    
    basicLongCondition_TTMS = (TTMS_Signals_TTMS == 1) if redGreen_TTMS else (TTMS_Signals_TTMS > 0)
    basicShortCondition_TTMS = (TTMS_Signals_TTMS == -1) if redGreen_TTMS else (TTMS_Signals_TTMS < 0)
    
    TTMS_SignalsLong_TTMS = np.where(highlightMovements_TTMS, NoSqz_TTMS & basicLongCondition_TTMS, basicLongCondition_TTMS)
    TTMS_SignalsShort_TTMS = np.where(highlightMovements_TTMS, NoSqz_TTMS & basicShortCondition_TTMS, basicShortCondition_TTMS)
    
    TTMS_SignalsLongCross_TTMS = (~TTMS_SignalsLong_TTMS.shift(1).astype(bool)) & TTMS_SignalsLong_TTMS.astype(bool) if cross_TTMS else TTMS_SignalsLong_TTMS.astype(bool)
    TTMS_SignalsShortCross_TTMS = (~TTMS_SignalsShort_TTMS.shift(1).astype(bool)) & TTMS_SignalsShort_TTMS.astype(bool) if cross_TTMS else TTMS_SignalsShort_TTMS.astype(bool)
    
    TTMS_SignalsLongFinal_TTMS = (~inverse_TTMS & TTMS_SignalsLongCross_TTMS.astype(bool)) | (inverse_TTMS & TTMS_SignalsShortCross_TTMS.astype(bool)) if use_TTMS else np.ones(n, dtype=bool)
    TTMS_SignalsShortFinal_TTMS = (~inverse_TTMS & TTMS_SignalsShortCross_TTMS.astype(bool)) | (inverse_TTMS & TTMS_SignalsLongCross_TTMS.astype(bool)) if use_TTMS else np.ones(n, dtype=bool)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # Trend MA Implementation
    # ═══════════════════════════════════════════════════════════════════════════════
    mma = symbolData.ewm(span=trendPeriod, adjust=False).mean()
    smma = mma.ewm(span=trendPeriod, adjust=False).mean()
    
    impetmma = mma.diff()
    impetsmma = smma.diff()
    divma = np.abs(mma - smma) / 0.01  # syminfo.mintick approximation
    
    averimpet = (impetmma + impetsmma) / (2 * 0.01)
    
    tdfRaw = divma * (averimpet ** 3)
    tdfAbsRaw = np.abs(tdfRaw)
    
    for i in range(1, min(3 * trendPeriod, n)):
        cand = np.abs(tdfRaw.shift(i))
        tdfAbsRaw = np.where(cand > tdfAbsRaw, cand, tdfAbsRaw)
    
    ratio = tdfRaw / (tdfAbsRaw + 1e-10)
    
    # Super Smoother
    def ssf2(src, length):
        arg = np.sqrt(2) * np.pi / length
        a1 = np.exp(-arg)
        b1 = 2 * a1 * np.cos(arg)
        c2 = b1
        c3 = -a1 ** 2
        c1 = 1 - c2 - c3
        ssf = np.zeros(len(src))
        for i in range(len(src)):
            if i == 0:
                ssf[i] = src.iloc[i]
            elif i == 1:
                ssf[i] = c1 * src.iloc[i] + c2 * src.iloc[i-1] + c3 * src.iloc[i-1]
            else:
                ssf[i] = c1 * src.iloc[i] + c2 * ssf[i-1] + c3 * ssf[i-2]
        return pd.Series(ssf, index=src.index)
    
    smooth = pd.Series(np.nan, index=df.index)
    
    if maType1 == 'SSF2':
        smooth = ssf2(ratio, maPeriod)
    elif maType1 == 'DEMA':
        ema1 = ratio.ewm(span=maPeriod, adjust=False).mean()
        ema2 = ema1.ewm(span=maPeriod, adjust=False).mean()
        smooth = 2 * ema1 - ema2
    elif maType1 == 'EMA':
        smooth = ratio.ewm(span=maPeriod, adjust=False).mean()
    elif maType1 == 'HMA':
        half_length = int(maPeriod / 2)
        wma1 = (2 * ratio.ewm(span=half_length, adjust=False).mean())
        wma2 = ratio.ewm(span=maPeriod, adjust=False).mean()
        smooth = (wma1 - wma2).ewm(span=int(np.sqrt(maPeriod)), adjust=False).mean()
    elif maType1 == 'LSMA':
        smooth = ratio.rolling(maPeriod).mean()
    elif maType1 == 'PWMA':
        weights = np.array([(maPeriod - i) ** 2 for i in range(maPeriod)])
        weights = weights / weights.sum()
        smooth = ratio.rolling(maPeriod).apply(lambda x: (x * weights).sum(), raw=True)
    elif maType1 == 'SMMA':
        smooth = ratio.ewm(span=maPeriod, adjust=False).mean()
    elif maType1 == 'SMA':
        smooth = ratio.rolling(maPeriod).mean()
    elif maType1 == 'SWMA':
        weights = np.array([np.sin(i * np.pi / (maPeriod + 1)) for i in range(maPeriod)])
        weights = weights / weights.sum()
        smooth = ratio.rolling(maPeriod).apply(lambda x: (x * weights).sum(), raw=True)
    elif maType1 == 'TEMA':
        ema1 = ratio.ewm(span=maPeriod, adjust=False).mean()
        ema2 = ema1.ewm(span=maPeriod, adjust=False).mean()
        ema3 = ema2.ewm(span=maPeriod, adjust=False).mean()
        smooth = 3 * (ema1 - ema2) + ema3
    elif maType1 == 'TMA':
        smooth = ratio.rolling(int(np.ceil(maPeriod / 2))).mean().rolling(int(np.floor(maPeriod / 2) + 1)).mean()
    elif maType1 == 'VWMA':
        smooth = (ratio * df['volume']).rolling(maPeriod).sum() / df['volume'].rolling(maPeriod).sum()
    elif maType1 == 'WMA':
        weights = np.array([i + 1 for i in range(maPeriod)])
        weights = weights / weights.sum()
        smooth = ratio.rolling(maPeriod).apply(lambda x: (x * weights).sum(), raw=True)
    elif maType1 == 'ZLEMA':
        lag = 1 if maPeriod <= 2 else int((maPeriod - 1) / 2)
        smooth = (ratio + ratio - ratio.shift(lag)).ewm(span=maPeriod, adjust=False).mean()
    elif maType1 == 'JMA':
        alpha = 0.45 * (maPeriod - 1) / (0.45 * (maPeriod - 1) + 2)
        e0 = np.zeros(n)
        e1 = np.zeros(n)
        e2 = np.zeros(n)
        out = np.zeros(n)
        for i in range(n):
            e0[i] = (1 - alpha) * ratio.iloc[i] + alpha * (e0[i-1] if i > 0 else ratio.iloc[i])
            e1[i] = (1 - alpha) * (ratio.iloc[i] - e0[i]) + alpha * (e1[i-1] if i > 0 else 0)
            e2[i] = (1 - alpha) ** 2 * (e0[i] + e1[i] - (out[i-1] if i > 0 else 0)) + alpha ** 2 * (e2[i-1] if i > 0 else 0)
            out[i] = e2[i] + (out[i-1] if i > 0 else 0)
        smooth = pd.Series(out, index=df.index)
    
    # Final entry conditions
    long_condition = signalLongE2PSSFinal & TTMS_SignalsLongFinal_TTMS & ~pd.isna(smooth) & ~pd.isna(mom_TTMS)
    short_condition = signalShortE2PSSFinal & TTMS_SignalsShortFinal_TTMS & ~pd.isna(smooth) & ~pd.isna(mom_TTMS)
    
    # Generate entries
    for i in range(n):
        if pd.isna(mom_TTMS.iloc[i]) or pd.isna(smooth.iloc[i]):
            continue
        
        entry_price = close.iloc[i]
        
        if long_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
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
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
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