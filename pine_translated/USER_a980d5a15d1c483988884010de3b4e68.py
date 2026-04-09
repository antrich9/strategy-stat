import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']
    
    # E2PSS parameters
    PeriodE2PSS = 15
    useE2PSS = True
    inverseE2PSS = False
    
    # E2PSS calculation
    a1 = np.exp(-1.414 * np.pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    Filt2 = np.zeros(len(df))
    TriggerE2PSS = np.zeros(len(df))
    
    for i in range(len(df)):
        if i < 1:
            Filt2[i] = close.iloc[i]
        else:
            Filt2[i] = coef1 * close.iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
        if i >= 1:
            TriggerE2PSS[i] = Filt2[i-1]
    
    # Signal generation
    signalLongE2PSS = Filt2 > TriggerE2PSS
    signalShortE2PSS = Filt2 < TriggerE2PSS
    
    if useE2PSS:
        signalLongE2PSSFinal = signalLongE2PSS if not inverseE2PSS else signalShortE2PSS
        signalShortE2PSSFinal = signalShortE2PSS if not inverseE2PSS else signalLongE2PSS
    else:
        signalLongE2PSSFinal = np.ones(len(df), dtype=bool)
        signalShortE2PSSFinal = np.ones(len(df), dtype=bool)
    
    # Trendilo parameters
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    pct_change = close.pct_change(trendilo_smooth) * 100
    avg_pct_change = pct_change.rolling(window=trendilo_length).apply(lambda x: np.sum(x) / trendilo_length, raw=True)
    rms = trendilo_bmult * np.sqrt((avg_pct_change ** 2).rolling(window=trendilo_length).mean())
    trendilo_dir = np.where(avg_pct_change > rms, 1, np.where(avg_pct_change < -rms, -1, 0))
    
    # TTM Squeeze parameters
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    BB_basis_TTMS = close.rolling(window=length_TTMS).mean()
    BB_std_TTMS = close.rolling(window=length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + BB_mult_TTMS * BB_std_TTMS
    BB_lower_TTMS = BB_basis_TTMS - BB_mult_TTMS * BB_std_TTMS
    
    KC_mult_TTMS = 1.5
    KC_basis_TTMS = close.rolling(window=length_TTMS).mean()
    KC_std_TTMS = low.rolling(window=length_TTMS).std()
    KC_upper_TTMS = KC_basis_TTMS + KC_mult_TTMS * KC_std_TTMS
    KC_lower_TTMS = KC_basis_TTMS - KC_mult_TTMS * KC_std_TTMS
    
    sqz_on = (BB_lower_TTMS > KC_lower_TTMS) & (BB_upper_TTMS < KC_upper_TTMS)
    
    return {
        'E2PSS_Filter': Filt2,
        'E2PSS_Trigger': TriggerE2PSS,
        'signal_long': signalLongE2PSSFinal,
        'signal_short': signalShortE2PSSFinal,
        'trendilo_dir': trendilo_dir,
        'BB_upper': BB_upper_TTMS,
        'BB_lower': BB_lower_TTMS,
        'KC_upper': KC_upper_TTMS,
        'KC_lower': KC_lower_TTMS,
        'squeeze_on': sqz_on
    }