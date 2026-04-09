import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Input parameters
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
    atrLength = 14
    
    # E2PSS
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    Filt2 = pd.Series(np.nan, index=df.index, dtype=float)
    TriggerE2PSS = pd.Series(np.nan, index=df.index, dtype=float)
    price_e2pss = (df['high'] + df['low']) / 2
    
    for i in range(len(df)):
        if i < 3:
            Filt2.iloc[i] = price_e2pss.iloc[i]
        else:
            Filt2.iloc[i] = coef1 * price_e2pss.iloc[i] + coef2 * Filt2.iloc[i-1] + coef3 * Filt2.iloc[i-2]
        if i == 0:
            TriggerE2PSS.iloc[i] = 0.0
        else:
            TriggerE2PSS.iloc[i] = Filt2.iloc[i-1]
    
    signalLongE2PSS = Filt2 > TriggerE2PSS
    signalShortE2PSS = Filt2 < TriggerE2PSS
    if useE2PSS:
        signalLongE2PSSFinal = signalLongE2PSS if not inverseE2PSS else signalShortE2PSS
        signalShortE2PSSFinal = signalShortE2PSS if not inverseE2PSS else signalLongE2PSS
    else:
        signalLongE2PSSFinal = pd.Series(True, index=df.index)
        signalShort