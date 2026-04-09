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
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # FVMA parameters
    ADX_Length_FVMA = 2
    Weighting_FVMA = 10.0
    MA_Length_FVMA = 6
    
    # Initialize FVMA arrays
    sPDI_FVMA = np.zeros(len(df))
    sMDI_FVMA = np.zeros(len(df))
    ADX_FVMA = np.zeros(len(df))
    STR_FVMA_arr = np.zeros(len(df))
    
    Hi_FVMA = high.values
    Hi1_FVMA = high.shift(1).values
    Lo_FVMA = low.values
    Lo1_FVMA = low.shift(1).values
    Close1_FVMA = close.shift(1).values
    
    Bulls1_FVMA = 0.5 * (np.abs(Hi_FVMA - Hi1_FVMA) + Hi_FVMA - Hi1_FVMA)
    Bears1_FVMA = 0.5 * (np.abs(Lo1_FVMA - Lo_FVMA) + Lo1_FVMA - Lo_FVMA)
    
    Bulls_FVMA = np.where(Bulls1_FVMA > Bears1_FVMA, 0, np.where(Bulls1_FVMA == Bears1_FVMA, 0, Bears1_FVMA))
    Bears_FVMA_arr = np.where(Bulls1_FVMA < Bears1_FVMA, 0, np.where(Bulls1_FVMA == Bears1_FVMA, 0, Bears1_FVMA))
    
    for i in range(1, len(df)):
        sPDI_FVMA[i] = (Weighting_FVMA * sPDI_FVMA[i-1] + Bulls_FVMA[i]) / (Weighting_FVMA + 1)
        sMDI_FVMA[i] = (Weighting_FVMA * sMDI_FVMA[i-1] + Bears_FVMA_arr[i]) / (Weighting_FVMA + 1)
        
        TR_FVMA = max(Hi_FVMA[i] - Lo_FVMA[i], Hi_FVMA[i] - Close1_FVMA[i])
        STR_FVMA_arr[i] = (Weighting_FVMA * STR_FVMA_arr[i-1] + TR_FVMA) / (Weighting_FVMA + 1)
        
        PDI_FVMA_val = sPDI_FVMA[i] / STR_FVMA_arr[i] if STR_FVMA_arr[i] > 0 else 0
        MDI_FVMA_val = sMDI_FVMA[i] / STR_FVMA_arr[i] if STR_FVMA_arr[i] > 0 else 0
        DX_FVMA_val = abs(PDI_FVMA_val - MDI_FVMA_val) / (PDI_FVMA_val + MDI_FVMA_val) if (PDI_FVMA_val + MDI_FVMA_val) > 0 else 0
        
        ADX_FVMA[i] = (Weighting_FVMA * ADX_FVMA[i-1] + DX_FVMA_val) / (Weighting_FVMA + 1)
    
    vADX_FVMA = ADX_FVMA
    adxlow_FVMA = pd.Series(ADX_FVMA).rolling(ADX_Length_FVMA).min().values
    adxmax_FVMA = pd.Series(ADX_FVMA).rolling(ADX_Length_FVMA).max().values
    ADXmin_FVMA = np.minimum(1000000.0, adxlow_FVMA)
    ADXmax_FVMA = np.maximum(-1.0, adxmax_FVMA)
    Diff_FVMA = ADXmax_FVMA - ADXmin_FVMA
    Const_FVMA = (vADX_FVMA - ADXmin_FVMA) / Diff_FVMA if Diff_FVMA > 0 else 0
    
    VarMA_FVMA = close.values.copy()
    for i in range(1, len(df)):
        VarMA_FVMA[i] = ((2 - Const_FVMA[i]) * VarMA_FVMA[i-1] + Const_FVMA[i] * close.values[i]) / 2
    
    MA_FVMA = pd.Series(VarMA_FVMA).rolling(MA_Length_FVMA).mean().values
    
    FVMASignals = np.where(MA_FVMA > np.roll(MA_FVMA, 1), 1, -1)
    FVMASignals[0] = -1
    
    basicLongCondition_FVMA = (FVMASignals > 0) & (close.values > MA_FVMA)
    basicShortCondition_FVMA = (FVMASignals < 0) & (close.values < MA_FVMA)
    
    useFVMA = True
    highlightMovementsFVMA_FVMA = True
    crossFVMA = True
    inverseFVMA = False
    
    FVMASignalsLong = useFVMA and (highlightMovementsFVMA_FVMA and basicLongCondition_FVMA or ~highlightMovementsFVMA_FVMA and close.values > MA_FVMA)
    FVMASignalsShort = useFVMA and (highlightMovementsFVMA_FVMA and basicShortCondition_FVMA or ~highlightMovementsFVMA_FVMA and close.values < MA_FVMA)
    
    FVMASignalsLong_arr = np.where(FVMASignalsLong, 1, 0)
    FVMASignalsShort_arr = np.where(FVMASignalsShort, 1, 0)
    
    FVMASignalsLongCross = np.where(crossFVMA, 
                                    (~np.roll(FVMASignalsLong_arr, 1).astype(bool)) & FVMASignalsLong_arr,
                                    FVMASignalsLong_arr)
    FVMASignalsLongCross[0] = 0
    
    FVMASignalsShortCross = np.where(crossFVMA,
                                      (~np.roll(FVMASignalsShort_arr, 1).astype(bool)) & FVMASignalsShort_arr,
                                      FVMASignalsShort_arr)
    FVMASignalsShortCross[0] = 0
    
    FVMASignalsLongFinal = np.where(inverseFVMA, FVMASignalsShortCross, FVMASignalsLongCross)
    FVMASignalsShortFinal = np.where(inverseFVMA, FVMASignalsLongCross, FVMASignalsShortCross)
    
    # Zero Lag MACD
    source = close
    fastLength = 12
    slowLength = 26
    signalLength = 9
    
    ema1 = source.ewm(span=fastLength, adjust=False).mean()
    ema2 = ema1.ewm(span=fastLength, adjust=False).mean()
    differenceFast = ema1 - ema2
    demaFast = ema1 + differenceFast
    
    emas1 = source.ewm(span=slowLength, adjust=False).mean()
    emas2 = emas1.ewm(span=slowLength, adjust=False).mean()
    differenceSlow = emas1 - emas2
    demaSlow = emas1 + differenceSlow
    
    ZeroLagMACD = demaFast - demaSlow
    
    emasig1 = ZeroLagMACD.ewm(span=signalLength, adjust=False).mean()
    emasig2 = emasig1.ewm(span=signalLength, adjust=False).mean()
    signal = emasig1 + (emasig1 - emasig2)
    
    # Supertrend
    Periods = 10
    src = (df['high'] + df['low']) / 2
    Multiplier = 3.0
    
    atr = pd.Series(index=df.index)
    tr = pd.concat([df['high'] - df['low'], 
                    (df['high'] - df['close'].shift(1)).abs(), 
                    (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
    
    alpha = 1.0 / Periods
    atr.values[0] = tr.values[0]
    for i in range(1, len(df)):
        atr.values[i] = (1 - alpha) * atr.values[i-1] + alpha * tr.values[i]
    
    up = src - Multiplier * atr
    dn = src + Multiplier * atr
    
    up_arr = up.values.copy()
    dn_arr = dn.values.copy()
    trend = np.ones(len(df))
    
    for i in range(1, len(df)):
        if close.values[i-1] > up_arr[i-1]:
            up_arr[i] = max(up_arr[i], up_arr[i-1])
        else:
            up_arr[i] = up.values[i]
            
        if close.values[i-1] < dn_arr[i-1]:
            dn_arr[i] = min(dn_arr[i], dn_arr[i-1])
        else:
            dn_arr[i] = dn.values[i]
    
    for i in range(1, len(df)):
        if trend[i-1] == -1 and close.values[i] > dn_arr[i-1]:
            trend[i] = 1
        elif trend[i-1] == 1 and close.values[i] < up_arr[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
    
    ema200 = close.ewm(span=200, adjust=False).mean()
    
    # Entry conditions
    MA_FVMA_series = pd.Series(MA_FVMA, index=df.index)
    
    entryLongCondition = (close > ema200) & (FVMASignalsLongFinal == 1) & (ZeroLagMACD > 0)
    notLateEntryLong = close.shift(1) < MA_FVMA_series
    
    longEntrySignal = entryLongCondition & notLateEntryLong
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if longEntrySignal.iloc[i] and not np.isnan(MA_FVMA[i]) and not np.isnan(ZeroLagMACD.iloc[i]):
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries