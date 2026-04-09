import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # --- Inputs ---
    weighting = 10.0
    ma_length = 6
    adx_length = 2
    fast_length = 12
    slow_length = 26
    signal_length = 9
    vfi_length = 130
    coef = 0.2
    vcoef = 2.5
    signal_length1 = 5
    adxlen = 14
    dilen = 14
    
    # --- FVMA Calculation ---
    Hi_FVMA = high
    Hi1_FVMA = high.shift(1)
    Lo_FVMA = low
    Lo1_FVMA = low.shift(1)
    Close1_FVMA = close.shift(1)
    
    Bulls1_FVMA = 0.5 * (np.abs(Hi_FVMA - Hi1_FVMA) + Hi_FVMA - Hi1_FVMA)
    Bears1_FVMA = 0.5 * (np.abs(Lo1_FVMA - Lo_FVMA) + Lo1_FVMA - Lo_FVMA)
    
    Bulls_FVMA = np.where(Bulls1_FVMA > Bears1_FVMA, 0, np.where(Bulls1_FVMA == Bears1_FVMA, 0, Bears1_FVMA))
    Bears_FVMA = np.where(Bulls1_FVMA < Bears1_FVMA, 0, np.where(Bulls1_FVMA == Bears1_FVMA, 0, Bulls1_FVMA))
    
    # sPDI_FVMA and sMDI_FVMA - Wilder smoothed
    alpha_dm = 1 / (weighting + 1)
    sPDI_FVMA = pd.Series(Bulls_FVMA).ewm(alpha=alpha_dm, adjust=False).mean()
    sMDI_FVMA = pd.Series(Bears_FVMA).ewm(alpha=alpha_dm, adjust=False).mean()
    
    # True Range components
    tr1 = high - low
    tr2 = high - Close1_FVMA
    tr3 = Close1_FVMA - low
    TR_FVMA_raw = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # STR_FVMA - Wilder smoothed TR
    STR_FVMA = pd.Series(TR_FVMA_raw).ewm(alpha=alpha_dm, adjust=False).mean()
    
    # PDI, MDI, DX
    PDI_FVMA = sPDI_FVMA / STR_FVMA
    MDI_FVMA = sMDI_FVMA / STR_FVMA
    PDI_FVMA = PDI_FVMA.fillna(0)
    MDI_FVMA = MDI_FVMA.fillna(0)
    DX_FVMA = np.abs(PDI_FVMA - MDI_FVMA) / (PDI_FVMA + MDI_FVMA)
    DX_FVMA = DX_FVMA.replace([np.inf, -np.inf], 0).fillna(0)
    
    # ADX_FVMA - Wilder smoothed DX
    ADX_FVMA = pd.Series(DX_FVMA.values).ewm(alpha=alpha_dm, adjust=False).mean()
    
    # Const_FVMA calculation
    adxlow_FVMA = ADX_FVMA.rolling(adx_length).min()
    adxmax_FVMA = ADX_FVMA.rolling(adx_length).max()
    ADXmin_FVMA = np.minimum(1000000.0, adxlow_FVMA)
    ADXmax_FVMA = np.maximum(-1.0, adxmax_FVMA)
    Diff_FVMA = ADXmax_FVMA - ADXmin_FVMA
    Const_FVMA = np.where(Diff_FVMA > 0, (ADX_FVMA - ADXmin_FVMA) / Diff_FVMA, 0)
    
    # VarMA_FVMA - using ewm with time-varying alpha
    alpha_var = Const_FVMA / 2
    alpha_var = alpha_var.replace([0, np.nan], 0.00001).fillna(0.00001)
    VarMA_FVMA = pd.Series(index=close.index)
    VarMA_FVMA.iloc[0] = close.iloc[0]
    for i in range(1, len(close)):
        a = alpha_var.iloc[i]
        VarMA_FVMA.iloc[i] = ((2 - a) * VarMA_FVMA.iloc[i-1] + a * close.iloc[i]) / 2
    VarMA_FVMA = pd.Series(VarMA_FVMA.values, index=close.index)
    
    # MA_FVMA - SMA of VarMA
    MA_FVMA = VarMA_FVMA.rolling(ma_length).mean()
    
    # FVMASignals
    FVMASignals = np.where(MA_FVMA > MA_FVMA.shift(1), 1, -1)
    
    # Basic conditions
    basicLongCondition_FVMA = (FVMASignals > 0) & (close > MA_FVMA)
    basicShortCondition_FVMA = (FVMASignals < 0) & (close < MA_FVMA)
    
    # --- Zero Lag MACD ---
    source = close
    ema1 = source.ewm(span=fast_length, adjust=False).mean()
    ema2 = ema1.ewm(span=fast_length, adjust=False).mean()
    differenceFast = ema1 - ema2
    demaFast = ema1 + differenceFast
    
    emas1 = source.ewm(span=slow_length, adjust=False).mean()
    emas2 = emas1.ewm(span=slow_length, adjust=False).mean()
    differenceSlow = emas1 - emas2
    demaSlow = emas1 + differenceSlow
    
    ZeroLagMACD = demaFast - demaSlow
    
    emasig1 = ZeroLagMACD.ewm(span=signal_length, adjust=False).mean()
    emasig2 = emasig1.ewm(span=signal_length, adjust=False).mean()
    signal = (2 * emasig1) - emasig2
    
    hist = ZeroLagMACD - signal
    
    macdSignal = (ZeroLagMACD > signal) & (hist > 0)
    
    longmacd = (ZeroLagMACD > signal) & (ZeroLagMACD.shift(1) <= signal.shift(1))
    shortmacd = (ZeroLagMACD < signal) & (ZeroLagMACD.shift(1) >= signal.shift(1))
    
    # --- VFI ---
    typical = (high + low + close) / 3
    inter = np.log(typical) - np.log(typical.shift(1))
    vinter = inter.rolling(30).std()
    cutoff = coef * vinter * close
    vave = volume.rolling(vfi_length).mean().shift(1)
    vmax = vave * vcoef
    vc = np.minimum(volume, vmax)
    mf = typical - typical.shift(1)
    vcp = np.where(mf > cutoff, vc, np.where(mf < -cutoff, -vc, 0))
    vfi_raw = pd.Series(vcp).rolling(vfi_length).sum() / vave
    vfi = vfi_raw.rolling(3).mean()
    vfima = vfi.ewm(span=signal_length1, adjust=False).mean()
    
    # --- ADX for entry ---
    up = high.diff()
    down = -low.diff()
    plusDM = np.where((up > down) & (up > 0), up, 0)
    minusDM = np.where((down > up) & (down > 0), down, 0)
    truerange = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - close.shift(1))), np.abs(low - close.shift(1)))).ewm(alpha=1/dilen, adjust=False).mean()
    plus = (pd.Series(plusDM).ewm(alpha=1/dilen, adjust=False).mean() / truerange) * 100
    minus = (pd.Series(minusDM).ewm(alpha=1/dilen, adjust=False).mean() / truerange) * 100
    plus = plus.fillna(0)
    minus = minus.fillna(0)
    sum_dir = plus + minus
    sum_dir = sum_dir.replace(0, 1)
    adx = (np.abs(plus - minus) / sum_dir) * 100
    sig = pd.Series(adx.values).ewm(alpha=1/adxlen, adjust=False).mean()
    adxCondition = sig > 20
    
    # --- Entry Conditions ---
    longCondition = basicLongCondition_FVMA & longmacd & (vfi > vfima) & adxCondition
    shortCondition = basicShortCondition_FVMA & shortmacd & (vfi < vfima) & adxCondition
    
    # --- Generate Entries ---
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(MA_FVMA.iloc[i]) or pd.isna(ZeroLagMACD.iloc[i]) or pd.isna(vfi.iloc[i]) or pd.isna(sig.iloc[i]):
            continue
        
        direction = None
        if longCondition.iloc[i]:
            direction = 'long'
        elif shortCondition.iloc[i]:
            direction = 'short'
        
        if direction:
            ts = int(df['time'].iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            }
            entries.append(entry)
            trade_num += 1
    
    return entries