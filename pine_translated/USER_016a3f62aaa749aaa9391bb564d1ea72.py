import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_col = df['open']
    
    # Metro RSI Parameters (using defaults from inputs)
    periodRSI_Metro = 14
    stepSizeFast_Metro = 5
    stepSizeSlow_Metro = 15
    useHeartbeat_Metro = True
    crossConfirm_Metro = True
    inverseHeartbeat_Metro = False
    
    # Wilders RSI for Metro
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/periodRSI_Metro, min_periods=periodRSI_Metro, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/periodRSI_Metro, min_periods=periodRSI_Metro, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsiLine_Metro = 100 - (100 / (1 + rs))
    
    # Step RSI Fast and Slow
    stepRSIFast_Metro = pd.Series(np.nan, index=df.index)
    stepRSISlow_Metro = pd.Series(np.nan, index=df.index)
    stepRSIFast_Metro.iloc[0] = rsiLine_Metro.iloc[0]
    stepRSISlow_Metro.iloc[0] = rsiLine_Metro.iloc[0]
    for i in range(1, len(df)):
        prev_fast = stepRSIFast_Metro.iloc[i-1]
        prev_slow = stepRSISlow_Metro.iloc[i-1]
        rsi_val = rsiLine_Metro.iloc[i]
        stepRSIFast_Metro.iloc[i] = max(min(prev_fast, rsi_val + stepSizeFast_Metro), rsi_val - stepSizeFast_Metro)
        stepRSISlow_Metro.iloc[i] = max(min(prev_slow, rsi_val + stepSizeSlow_Metro), rsi_val - stepSizeSlow_Metro)
    
    # Metro heartbeat signals
    basicLongCondition_Metro = stepRSIFast_Metro > stepRSISlow_Metro
    basicShortCondition_Metro = stepRSIFast_Metro < stepRSISlow_Metro
    
    heartbeatLong_Metro = basicLongCondition_Metro if useHeartbeat_Metro and not inverseHeartbeat_Metro else (basicShortCondition_Metro if useHeartbeat_Metro else False)
    heartbeatShort_Metro = basicShortCondition_Metro if useHeartbeat_Metro and not inverseHeartbeat_Metro else (basicLongCondition_Metro if useHeartbeat_Metro else False)
    
    heartbeatLongFinal_Metro = heartbeatLong_Metro & ~(heartbeatLong_Metro.shift(1).fillna(False)) if crossConfirm_Metro else heartbeatLong_Metro
    heartbeatShortFinal_Metro = heartbeatShort_Metro & ~(heartbeatShort_Metro.shift(1).fillna(False)) if crossConfirm_Metro else heartbeatShort_Metro
    
    # FVMA Parameters
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
    Bears_FVMA = np.where(Bulls1_FVMA > Bears1_FVMA, 0, np.where(Bulls1_FVMA == Bears1_FVMA, 0, Bears1_FVMA))
    Bulls_FVMA = np.where(Bulls1_FVMA < Bears1_FVMA, 0, np.where(Bulls1_FVMA == Bears1_FVMA, 0, Bulls1_FVMA))
    
    TR_FVMA = np.maximum(Hi_FVMA - Lo_FVMA, np.maximum(np.abs(Hi_FVMA - Close1_FVMA), np.abs(Lo_FVMA - Close1_FVMA)))
    
    sPDI_FVMA = pd.Series(0.0, index=df.index)
    sMDI_FVMA = pd.Series(0.0, index=df.index)
    STR_FVMA = pd.Series(np.nan, index=df.index)
    ADX_FVMA = pd.Series(0.0, index=df.index)
    
    for i in range(1, len(df)):
        sPDI_FVMA.iloc[i] = (Weighting_FVMA * sPDI_FVMA.iloc[i-1] + Bulls_FVMA.iloc[i]) / (Weighting_FVMA + 1)
        sMDI_FVMA.iloc[i] = (Weighting_FVMA * sMDI_FVMA.iloc[i-1] + Bears_FVMA.iloc[i]) / (Weighting_FVMA + 1)
        STR_FVMA.iloc[i] = (Weighting_FVMA * STR_FVMA.iloc[i-1] + TR_FVMA.iloc[i]) / (Weighting_FVMA + 1)
        PDI_FVMA_val = sPDI_FVMA.iloc[i] / STR_FVMA.iloc[i] if STR_FVMA.iloc[i] > 0 else 0
        MDI_FVMA_val = sMDI_FVMA.iloc[i] / STR_FVMA.iloc[i] if STR_FVMA.iloc[i] > 0 else 0
        DX_FVMA_val = np.abs(PDI_FVMA_val - MDI_FVMA_val) / (PDI_FVMA_val + MDI_FVMA_val) if (PDI_FVMA_val + MDI_FVMA_val) > 0 else 0
        ADX_FVMA.iloc[i] = (Weighting_FVMA * ADX_FVMA.iloc[i-1] + DX_FVMA_val) / (Weighting_FVMA + 1)
    
    adxlow_FVMA = ADX_FVMA.rolling(ADX_Length_FVMA).min()
    adxmax_FVMA = ADX_FVMA.rolling(ADX_Length_FVMA).max()
    ADXmin_FVMA = np.minimum(1000000.0, adxlow_FVMA)
    ADXmax_FVMA = np.maximum(-1.0, adxmax_FVMA)
    Diff_FVMA = ADXmax_FVMA - ADXmin_FVMA
    Const_FVMA = np.where(Diff_FVMA > 0, (ADX_FVMA - ADXmin_FVMA) / Diff_FVMA, 0)
    
    VarMA_FVMA = pd.Series(np.nan, index=df.index)
    VarMA_FVMA.iloc[0] = close.iloc[0]
    for i in range(1, len(df)):
        VarMA_FVMA.iloc[i] = ((2 - Const_FVMA.iloc[i]) * VarMA_FVMA.iloc[i-1] + Const_FVMA.iloc[i] * close.iloc[i]) / 2
    
    MA_FVMA = VarMA_FVMA.rolling(MA_Length_FVMA).mean()
    FVMASignals = pd.Series(1, index=df.index)
    FVMASignals[MA_FVMA < MA_FVMA.shift(1)] = -1
    
    basicLongCondition_FVMA = (FVMASignals > 0) & (close > MA_FVMA)
    basicShortCondition_FVMA = (FVMASignals < 0) & (close < MA_FVMA)
    
    FVMASignalsLong = basicLongCondition_FVMA if useFVMA else pd.Series(True, index=df.index)
    FVMASignalsShort = basicShortCondition_FVMA if useFVMA else pd.Series(True, index=df.index)
    
    FVMASignalsLongCross = FVMASignalsLong & ~(FVMASignalsLong.shift(1).fillna(False)) if crossFVMA else FVMASignalsLong
    FVMASignalsShortCross = FVMASignalsShort & ~(FVMASignalsShort.shift(1).fillna(False)) if crossFVMA else FVMASignalsShort
    
    FVMASignalsLongFinal = FVMASignalsShortCross if inverseFVMA else FVMASignalsLongCross
    FVMASignalsShortFinal = FVMASignalsLongCross if inverseFVMA else FVMASignalsShortCross
    
    # VFI Parameters
    vfilength = 130
    coef = 0.2
    vcoef = 2.5
    signalLength = 5
    
    typical = (high + low + close) / 3
    inter = np.log(typical) - np.log(typical.shift(1))
    vinter = inter.rolling(30).std()
    cutoff = coef * vinter * close
    vave = volume.shift(1).rolling(vfilength).mean()
    vmax = vave * vcoef
    vc = np.minimum(volume, vmax)
    mf = typical - typical.shift(1)
    vcp = np.where(mf > cutoff, vc, np.where(mf < -cutoff, -vc, 0))
    vfi = (vcp.rolling(vfilength).sum() / vave).rolling(3).mean()
    vfima = vfi.ewm(span=signalLength, adjust=False).mean()
    
    # SuperTrend Parameters
    length_ST = 22
    mult_ST = 3.0
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    atr_ST = tr.ewm(alpha=1/length_ST, min_periods=length_ST, adjust=False).mean()
    atrValue_ST = atr_ST * mult_ST
    
    src_ST = (high + low) / 2
    longStop = src_ST - atrValue_ST
    shortStop = src_ST + atrValue_ST
    
    dir_ST = pd.Series(1, index=df.index)
    dir_ST.iloc[0] = 1
    
    for i in range(1, len(df)):
        prev_long_stop = longStop.iloc[i-1]
        prev_short_stop = shortStop.iloc[i-1]
        
        new_long = src_ST.iloc[i] - atrValue_ST.iloc[i]
        new_short = src_ST.iloc[i] + atrValue_ST.iloc[i]
        
        longStop.iloc[i] = new_long if new_long > 0 else prev_long_stop
        shortStop.iloc[i] = new_short if new_short > 0 else prev_short_stop
        
        if dir_ST.iloc[i-1] == -1 and high.iloc[i] > prev_short_stop:
            dir_ST.iloc[i] = 1
        elif dir_ST.iloc[i-1] == 1 and low.iloc[i] < prev_long_stop:
            dir_ST.iloc[i] = -1
    
    # Entry conditions
    long_entry_cond = (stepRSIFast_Metro > stepRSISlow_Metro) & FVMASignalsLongFinal & (vfi > vfima) & (dir_ST == 1)
    short_entry_cond = (stepRSIFast_Metro < stepRSISlow_Metro) & FVMASignalsShortFinal & (vfi < vfima) & (dir_ST == -1)
    
    long_entry = long_entry_cond & ~(long_entry_cond.shift(1).fillna(False))
    short_entry = short_entry_cond & ~(short_entry_cond.shift(1).fillna(False))
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = close.iloc[i]
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
        elif short_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = close.iloc[i]
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
    
    return entries