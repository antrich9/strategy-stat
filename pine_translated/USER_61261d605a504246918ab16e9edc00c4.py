import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    high = df['high']
    low = df['low']
    close = df['close']
    open_price = df['open']
    volume = df['volume']
    
    # Parameters (default values from Pine Script)
    volumeIndicator = "Volume Oscillator"
    gkLength = 14
    gkThreshold = 0.5
    vqLength = 14
    vqSignalLen = 5
    voShortLen = 5
    voLongLen = 10
    voThreshold = 0.0
    jvbLength = 14
    jvbMult = 2.0
    jvbJMALen = 7
    jvbPhase = 0
    stfLength = 60
    stfSmoothLen = 3
    stfThreshold = 90
    nvLength = 50
    nvThreshold = 1.0
    adxLength = 14
    adxSmoothing = 14
    adxThreshold = 25
    adxMAType = "EMA"
    adxMALen = 50
    atrbLength = 14
    atrbMult = 1.5
    atrbMALen = 20
    ttmsLength = 20
    ttmsBBMult = 2.0
    ttmsKCMult = 1.5
    ttmsMomLen = 12
    
    # EMAs for entry triggers
    fastEMA = close.ewm(span=8, adjust=False).mean()
    slowEMA = close.ewm(span=21, adjust=False).mean()
    
    # ATR (Wilder)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # JMA approximation
    def jma(src, length, phase):
        phase_ratio = max(0.5, min(2.5, phase / 100 + 1.5))
        beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
        alpha = beta ** phase_ratio
        jma_vals = np.zeros(len(src))
        jma_vals[0] = src.iloc[0]
        for i in range(1, len(src)):
            jma_vals[i] = (1 - alpha) * src.iloc[i] + alpha * jma_vals[i-1]
        return pd.Series(jma_vals, index=src.index)
    
    # [1] Garman-Klass Volatility
    logHL = np.log(high / low)
    logCO = np.log(close / open_price)
    gk = 0.5 * logHL * logHL - (2 * np.log(2) - 1) * logCO * logCO
    gk_val = gk.rolling(gkLength).mean() * 100
    gk_bull = gk_val > gkThreshold
    gk_bear = gk_val > gkThreshold
    
    # [2] Volatility Quality
    trueHigh = pd.concat([high, close.shift(1)], axis=1).max(axis=1)
    trueLow = pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    vqRaw = np.where(close > close.shift(1), trueHigh - trueLow, -(trueHigh - trueLow))
    vq_line = pd.Series(vqRaw, index=close.index).rolling(vqLength).mean()
    vq_sig = vq_line.rolling(vqSignalLen).mean()
    vq_bull = (vq_line > 0) & (vq_line > vq_sig)
    vq_bear = (vq_line < 0) & (vq_line < vq_sig)
    
    # [3] Volume Oscillator
    vo_short = volume.ewm(span=voShortLen, adjust=False).mean()
    vo_long = volume.ewm(span=voLongLen, adjust=False).mean()
    vo_val = (vo_short - vo_long) / vo_long * 100
    vo_bull = vo_val > voThreshold
    vo_bear = vo_val > voThreshold
    
    # [4] Jurik Volatility Bands
    jvb_mid = jma(close, jvbJMALen, jvbPhase)
    jvb_atr = atr
    jvb_upper = jvb_mid + jvb_atr * jvbMult
    jvb_lower = jvb_mid - jvb_atr * jvbMult
    jvb_bull = (close > jvb_mid) & (jvb_atr > jvb_atr.shift(1))
    jvb_bear = (close < jvb_mid) & (jvb_atr > jvb_atr.shift(1))
    
    # [5] Stiffness Indicator