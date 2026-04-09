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
    
    n = len(df)
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # ===== METRO RSI CALCULATIONS =====
    rsiType_Metro = "Regular"
    periodRSI_Metro = 14
    stepSizeFast_Metro = 5
    stepSizeSlow_Metro = 15
    overSold_Metro = 10
    overBought_Metro = 90
    
    useHeartbeat_Metro = True
    crossConfirm_Metro = True
    inverseHeartbeat_Metro = False
    
    # Calculate Wilders RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/periodRSI_Metro, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/periodRSI_Metro, adjust=False).mean()
    rs = avg_gain / avg_loss
    wilders_rsi = 100 - (100 / (1 + rs))
    
    # Use Wilders RSI (as per default logic)
    rsiLine_Metro = wilders_rsi
    
    # StepRSI Fast Line
    stepRSIFast_Metro = pd.Series(np.nan, index=df.index)
    stepRSIFast_Metro.iloc[0] = rsiLine_Metro.iloc[0]
    for i in range(1, n):
        prev_val = stepRSIFast_Metro.iloc[i-1]
        curr_rsi = rsiLine_Metro.iloc[i]
        stepRSIFast_Metro.iloc[i] = max(min(prev_val, curr_rsi + stepSizeFast_Metro), curr_rsi - stepSizeFast_Metro)
    
    # StepRSI Slow Line
    stepRSISlow_Metro = pd.Series(np.nan, index=df.index)
    stepRSISlow_Metro.iloc[0] = rsiLine_Metro.iloc[0]
    for i in range(1, n):
        prev_val = stepRSISlow_Metro.iloc[i-1]
        curr_rsi = rsiLine_Metro.iloc[i]
        stepRSISlow_Metro.iloc[i] = max(min(prev_val, curr_rsi + stepSizeSlow_Metro), curr_rsi - stepSizeSlow_Metro)
    
    # Basic signals
    basicLongCondition_Metro = stepRSIFast_Metro > stepRSISlow_Metro
    basicShortCondition_Metro = stepRSIFast_Metro < stepRSISlow_Metro
    
    heartbeatLong_Metro = useHeartbeat_Metro and (inverseHeartbeat_Metro if True else basicLongCondition_Metro)
    heartbeatShort_Metro = useHeartbeat_Metro and (inverseHeartbeat_Metro if True else basicShortCondition_Metro)
    
    heartbeatLongFinal_Metro = heartbeatLong_Metro & (~heartbeatLong_Metro.shift(1).fillna(False)) if crossConfirm_Metro else heartbeatLong_Metro
    heartbeatShortFinal_Metro = heartbeatShort_Metro & (~heartbeatShort_Metro.shift(1).fillna(False)) if crossConfirm_Metro else heartbeatShort_Metro
    
    # ===== FVMA CALCULATIONS =====
    ADX_Length_FVMA = 2
    Weighting_FVMA = 10.0
    MA_Length_FVMA = 6
    useFVMA = True
    crossFVMA = True
    inverseFVMA = False
    highlightMovementsFVMA_FVMA = True
    
    Hi_FVMA = high
    Hi1_FVMA = high.shift(1)
    Lo_FVMA = low
    Lo1_FVMA = low.shift(1)
    Close1_FVMA = close.shift(1)
    
    Bulls1_FVMA = 0.5 * (np.abs(Hi_FVMA - Hi1_FVMA) + Hi_FVMA - Hi1_FVMA)
    Bears1_FVMA = 0.5 * (np.abs(Lo1_FVMA - Lo_FVMA) + Lo1_FVMA - Lo_FVMA)
    
    Bulls_FVMA = np.where((Bulls1_FVMA > Bears1_FVMA), 0, 
                         np.where((Bulls1_FVMA == Bears1_FVMA), 0, Bulls1_FVMA))
    Bears_FVMA = np.where((Bulls1_FVMA > Bears1_FVMA), 0, 
                          np.where((Bulls1_FVMA == Bears1_FVMA), 0, Bears1_FVMA))
    
    sPDI_FVMA = pd.Series(0.0, index=df.index)
    sMDI_FVMA = pd.Series(0.0, index=df.index)
    STR_FVMA = pd.Series(np.nan, index=df.index)
    STR_FVMA.iloc[0] = high.iloc[0] - low.iloc[0]
    
    TR_FVMA = np.maximum(high - low, np.maximum(high - close.shift(1), close.shift(1) - low))
    
    for i in range(1, n):
        sPDI_FVMA.iloc[i] = (Weighting_FVMA * sPDI_FVMA.iloc[i-1] + Bulls_FVMA.iloc[i]) / (Weighting_FVMA + 1)
        sMDI_FVMA.iloc[i] = (Weighting_FVMA * sMDI_FVMA.iloc[i-1] + Bears_FVMA.iloc[i]) / (Weighting_FVMA + 1)
        STR_FVMA.iloc[i] = (Weighting_FVMA * STR_FVMA.iloc[i-1] + TR_FVMA.iloc[i]) / (Weighting_FVMA + 1)
    
    PDI_FVMA = np.where(STR_FVMA > 0, sPDI_FVMA / STR_FVMA, 0)
    MDI_FVMA = np.where(STR_FVMA > 0, sMDI_FVMA / STR_FVMA, 0)
    DX_FVMA = np.where(PDI_FVMA + MDI_FVMA > 0, np.abs(PDI_FVMA - MDI_FVMA) / (PDI_FVMA + MDI_FVMA), 0)
    
    ADX_FVMA = pd.Series(0.0, index=df.index)
    for i in range(1, n):
        ADX_FVMA.iloc[i] = (Weighting_FVMA * ADX_FVMA.iloc[i-1] + DX_FVMA.iloc[i]) / (Weighting_FVMA + 1)
    
    adxlow_FVMA = ADX_FVMA.rolling(ADX_Length_FVMA).min()
    adxmax_FVMA = ADX_FVMA.rolling(ADX_Length_FVMA).max()
    ADXmin_FVMA = np.minimum(1000000.0, adxlow_FVMA)
    ADXmax_FVMA = np.maximum(-1.0, adxmax_FVMA)
    Diff_FVMA = ADXmax_FVMA - ADXmin_FVMA
    Const_FVMA = np.where(Diff_FVMA > 0, (ADX_FVMA - ADXmin_FVMA) / Diff_FVMA, 0)
    
    VarMA_FVMA = pd.Series(np.nan, index=df.index)
    VarMA_FVMA.iloc[0] = close.iloc[0]
    for i in range(1, n):
        VarMA_FVMA.iloc[i] = ((2 - Const_FVMA.iloc[i]) * VarMA_FVMA.iloc[i-1] + Const_FVMA.iloc[i] * close.iloc[i]) / 2
    
    MA_FVMA = VarMA_FVMA.rolling(MA_Length_FVMA).mean()
    
    FVMASignals = np.where(MA_FVMA > MA_FVMA.shift(1), 1, -1)
    
    basicLongCondition_FVMA = (FVMASignals > 0) & (close > MA_FVMA)
    basicShortCondition_FVMA = (FVMASignals < 0) & (close < MA_FVMA)
    
    FVMASignalsLong = basicLongCondition_FVMA if useFVMA and highlightMovementsFVMA_FVMA else (close > MA_FVMA if useFVMA else True)
    FVMASignalsShort = basicShortCondition_FVMA if useFVMA and highlightMovementsFVMA_FVMA else (close < MA_FVMA if useFVMA else True)
    
    FVMASignalsLong_df = pd.Series(FVMASignalsLong, index=df.index)
    FVMASignalsShort_df = pd.Series(FVMASignalsShort, index=df.index)
    
    FVMASignalsLongCross = FVMASignalsLong_df & (~FVMASignalsLong_df.shift(1).fillna(False)) if crossFVMA else FVMASignalsLong_df
    FVMASignalsShortCross = FVMASignalsShort_df & (~FVMASignalsShort_df.shift(1).fillna(False)) if crossFVMA else FVMASignalsShort_df
    
    FVMASignalsLongFinal = FVMASignalsShortCross if inverseFVMA else FVMASignalsLongCross
    FVMASignalsShortFinal = FVMASignalsLongCross if inverseFVMA else FVMASignalsShortCross
    
    # ===== VFI CALCULATIONS =====
    vfilength = 130
    coef = 0.2
    vcoef = 2.5
    signalLength = 5
    smoothVFI = False
    
    typical = (high + low + close) / 3
    inter = np.log(typical) - np.log(typical.shift(1))
    vinter = inter.rolling(30).std()
    cutoff = coef * vinter * close
    vave = volume.rolling(vfilength).mean().shift(1)
    vmax = vave * vcoef
    vc = np.minimum(volume, vmax)
    mf = typical - typical.shift(1)
    vcp = np.where(mf > cutoff, vc, np.where(mf < -cutoff, -vc, 0))
    vfi_raw = pd.Series(np.nan, index=df.index)
    for i in range(vfilength, n):
        vfi_raw.iloc[i] = vcp.iloc[i-vfilength+1:i+1].sum() / vave.iloc[i] if vave.iloc[i] != 0 else 0
    vfi = vfi_raw.rolling(3).mean() if smoothVFI else vfi_raw
    vfima = vfi.ewm(span=signalLength, adjust=False).mean()
    
    # VFI crossover and crossunder
    vfi_crossover = (vfi > vfima) & (vfi.shift(1) <= vfima)
    vfi_crossunder = (vfi < vfima) & (vfi.shift(1) >= vfima)
    
    # ===== SUPERTREND CALCULATIONS =====
    length = 22
    mult = 3.0
    src = (high + low) / 2
    wicks = True
    
    highPrice = high if wicks else close
    lowPrice = low if wicks else close
    doji4price = (open == close) & (open == low) & (open == high)
    
    atrValue = ta_wilder_atr(high, low, close, length) * mult
    
    longStop = pd.Series(np.nan, index=df.index)
    shortStop = pd.Series(np.nan, index=df.index)
    dir = pd.Series(1, index=df.index)
    
    longStop.iloc[0] = src.iloc[0] - atrValue.iloc[0]
    shortStop.iloc[0] = src.iloc[0] + atrValue.iloc[0]
    
    for i in range(1, n):
        if doji4price.iloc[i]:
            longStop.iloc[i] = longStop.iloc[i-1]
            shortStop.iloc[i] = shortStop.iloc[i-1]
        else:
            longStop.iloc[i] = src.iloc[i] - atrValue.iloc[i] if src.iloc[i] - atrValue.iloc[i] > 0 else longStop.iloc[i-1]
            longStop.iloc[i] = lowPrice.iloc[i-1] > longStop.iloc[i-1] if True else max(longStop.iloc[i], longStop.iloc[i-1])
            
            shortStop.iloc[i] = src.iloc[i] + atrValue.iloc[i] if src.iloc[i] + atrValue.iloc[i] > 0 else shortStop.iloc[i-1]
            shortStop.iloc[i] = highPrice.iloc[i-1] < shortStop.iloc[i-1] if True else min(shortStop.iloc[i], shortStop.iloc[i-1])
        
        if dir.iloc[i-1] == -1 and highPrice.iloc[i] > shortStop.iloc[i-1]:
            dir.iloc[i] = 1
        elif dir.iloc[i-1] == 1 and lowPrice.iloc[i] < longStop.iloc[i-1]:
            dir.iloc[i] = -1
        else:
            dir.iloc[i] = dir.iloc[i-1]
    
    # Recalculate longStop and shortStop properly
    longStop = pd.Series(np.nan, index=df.index)
    shortStop = pd.Series(np.nan, index=df.index)
    dir = pd.Series(1, index=df.index)
    
    for i in range(n):
        if i == 0:
            longStop.iloc[i] = src.iloc[i] - atrValue.iloc[i]
            shortStop.iloc[i] = src.iloc[i] + atrValue.iloc[i]
        else:
            if doji4price.iloc[i]:
                longStop.iloc[i] = longStop.iloc[i-1]
                shortStop.iloc[i] = shortStop.iloc[i-1]
            else:
                candidate_long = src.iloc[i] - atrValue.iloc[i] if src.iloc[i] - atrValue.iloc[i] > 0 else longStop.iloc[i-1]
                longStop.iloc[i] = candidate_long
                
                candidate_short = src.iloc[i] + atrValue.iloc[i] if src.iloc[i] + atrValue.iloc[i] > 0 else shortStop.iloc[i-1]
                shortStop.iloc[i] = candidate_short
                
                if dir.iloc[i-1] == -1 and highPrice.iloc[i] > shortStop.iloc[i-1]:
                    dir.iloc[i] = 1
                elif dir.iloc[i-1] == 1 and lowPrice.iloc[i] < longStop.iloc[i-1]:
                    dir.iloc[i] = -1
                else:
                    dir.iloc[i] = dir.iloc[i-1]
    
    # Recalculate stops with proper logic
    longStop = pd.Series(np.nan, index=df.index)
    shortStop = pd.Series(np.nan, index=df.index)
    dir_st = pd.Series(1, index=df.index)
    
    for i in range(n):
        if i == 0:
            longStop.iloc[i] = src.iloc[i] - atrValue.iloc[i]
            shortStop.iloc[i] = src.iloc[i] + atrValue.iloc[i]
        else:
            if doji4price.iloc[i]:
                longStop.iloc[i] = longStop.iloc[i-1]
                shortStop.iloc[i] = shortStop.iloc[i-1]
                dir_st.iloc[i] = dir_st.iloc[i-1]
            else:
                candidate_long = src.iloc[i] - atrValue.iloc[i] if src.iloc[i] - atrValue.iloc[i] > 0 else longStop.iloc[i-1]
                longStop.iloc[i] = lowPrice.iloc[i-1] > longStop.iloc[i-1] if True else max(longStop.iloc[i], longStop.iloc[i-1])
                if lowPrice.iloc[i-1] > longStop.iloc[i-1]:
                    longStop.iloc[i] = max(longStop.iloc[i], longStop.iloc[i-1])
                
                candidate_short = src.iloc[i] + atrValue.iloc[i] if src.iloc[i] + atrValue.iloc[i] > 0 else shortStop.iloc[i-1]
                shortStop.iloc[i] = candidate_short
                if highPrice.iloc[i-1] < shortStop.iloc[i-1]:
                    shortStop.iloc[i] = min(shortStop.iloc[i], shortStop.iloc[i-1])
                
                if dir_st.iloc[i-1] == -1 and highPrice.iloc[i] > shortStop.iloc[i-1]:
                    dir_st.iloc[i] = 1
                elif dir_st.iloc[i-1] == 1 and lowPrice.iloc[i] < longStop.iloc[i-1]:
                    dir_st.iloc[i] = -1
                else:
                    dir_st.iloc[i] = dir_st.iloc[i-1]
    
    # Final entry conditions based on commented code
    allBullishCondition = (stepRSIFast_Metro > stepRSISlow_Metro) & FVMASignalsLongFinal & vfi_crossover & (dir_st == 1)
    allBearishCondition = (stepRSIFast_Metro < stepRSISlow_Metro) & FVMASignalsShortFinal & vfi_crossunder & (dir_st == -1)
    
    # Build entries
    entries = []
    trade_num = 1
    
    for i in range(1, n):
        ts = df['time'].iloc[i]
        
        if allBullishCondition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
        
        if allBearishCondition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
    
    return entries


def ta_wilder_atr(high, low, close, length):
    """Calculate Wilder's ATR"""
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.Series(np.maximum(np.maximum(tr1, tr2), tr3))
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr