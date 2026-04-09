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
    useFVMA = True
    crossFVMA = True
    inverseFVMA = False
    highlightMovementsFVMA_FVMA = True
    ADX_Length_FVMA = 2
    Weighting_FVMA = 10.0
    MA_Length_FVMA = 6
    
    # Initialize series
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Bulls1 and Bears1
    Hi_FVMA = high
    Hi1_FVMA = high.shift(1)
    Lo_FVMA = low
    Lo1_FVMA = low.shift(1)
    Close1_FVMA = close.shift(1)
    
    Bulls1_FVMA = 0.5 * (np.abs(Hi_FVMA - Hi1_FVMA) + Hi_FVMA - Hi1_FVMA)
    Bears1_FVMA = 0.5 * (np.abs(Lo1_FVMA - Lo_FVMA) + Lo1_FVMA - Lo_FVMA)
    
    Bulls_FVMA = np.where(Bulls1_FVMA > Bears1_FVMA, 0, np.where(Bulls1_FVMA == Bears1_FVMA, 0, Bears1_FVMA))
    Bears_FVMA = np.where(Bulls1_FVMA < Bears1_FVMA, 0, np.where(Bulls1_FVMA == Bears1_FVMA, 0, Bulls1_FVMA))
    
    # Weighted moving average for sPDI and sMDI
    sPDI_FVMA = pd.Series(0.0, index=df.index)
    sMDI_FVMA = pd.Series(0.0, index=df.index)
    
    # True Range
    TR_FVMA = np.maximum(Hi_FVMA - Lo_FVMA, Hi_FVMA - Close1_FVMA)
    
    STR_FVMA = pd.Series(0.0, index=df.index)
    ADX_FVMA = pd.Series(0.0, index=df.index)
    
    # Calculate from bar 1 onwards
    for i in range(1, len(df)):
        sPDI_FVMA.iloc[i] = (Weighting_FVMA * sPDI_FVMA.iloc[i-1] + Bulls_FVMA.iloc[i]) / (Weighting_FVMA + 1)
        sMDI_FVMA.iloc[i] = (Weighting_FVMA * sMDI_FVMA.iloc[i-1] + Bears_FVMA.iloc[i]) / (Weighting_FVMA + 1)
        STR_FVMA.iloc[i] = (Weighting_FVMA * STR_FVMA.iloc[i-1] + TR_FVMA.iloc[i]) / (Weighting_FVMA + 1)
        
        PDI_FVMA_val = sPDI_FVMA.iloc[i] / STR_FVMA.iloc[i] if STR_FVMA.iloc[i] > 0 else 0
        MDI_FVMA_val = sMDI_FVMA.iloc[i] / STR_FVMA.iloc[i] if STR_FVMA.iloc[i] > 0 else 0
        DX_FVMA_val = np.abs(PDI_FVMA_val - MDI_FVMA_val) / (PDI_FVMA_val + MDI_FVMA_val) if (PDI_FVMA_val + MDI_FVMA_val) > 0 else 0
        
        ADX_FVMA.iloc[i] = (Weighting_FVMA * ADX_FVMA.iloc[i-1] + DX_FVMA_val) / (Weighting_FVMA + 1)
    
    # vADX_FVMA
    vADX_FVMA = ADX_FVMA
    
    # adxlow and adxmax using rolling min and max
    adxlow_FVMA = ADX_FVMA.rolling(window=ADX_Length_FVMA, min_periods=1).min()
    adxmax_FVMA = ADX_FVMA.rolling(window=ADX_Length_FVMA, min_periods=1).max()
    
    ADXmin_FVMA = np.minimum(1000000.0, adxlow_FVMA)
    ADXmax_FVMA = np.maximum(-1.0, adxmax_FVMA)
    Diff_FVMA = ADXmax_FVMA - ADXmin_FVMA
    Const_FVMA = np.where(Diff_FVMA > 0, (vADX_FVMA - ADXmin_FVMA) / Diff_FVMA, 0)
    
    # VarMA_FVMA
    VarMA_FVMA = pd.Series(close.values, index=df.index)
    
    for i in range(1, len(df)):
        VarMA_FVMA.iloc[i] = ((2 - Const_FVMA.iloc[i]) * VarMA_FVMA.iloc[i-1] + Const_FVMA.iloc[i] * close.iloc[i]) / 2
    
    # MA_FVMA
    MA_FVMA = VarMA_FVMA.rolling(window=MA_Length_FVMA, min_periods=1).mean()
    
    # FVMASignals
    FVMASignals = np.where(MA_FVMA > MA_FVMA.shift(1), 1, -1)
    
    # basic conditions
    basicLongCondition_FVMA = (FVMASignals > 0) & (close > MA_FVMA)
    basicShortCondition_FVMA = (FVMASignals < 0) & (close < MA_FVMA)
    
    # FVMASignalsLong and Short
    FVMASignalsLong = np.where(useFVMA, np.where(highlightMovementsFVMA_FVMA, basicLongCondition_FVMA, close > MA_FVMA), True)
    FVMASignalsShort = np.where(useFVMA, np.where(highlightMovementsFVMA_FVMA, basicShortCondition_FVMA, close < MA_FVMA), True)
    
    # Cross conditions
    FVMASignalsLongCross = np.where(crossFVMA, (~FVMASignalsLong.shift(1).fillna(False)) & FVMASignalsLong, FVMASignalsLong)
    FVMASignalsShortCross = np.where(crossFVMA, (~FVMASignalsShort.shift(1).fillna(False)) & FVMASignalsShort, FVMASignalsShort)
    
    # Final signals
    FVMASignalsLongFinal = np.where(inverseFVMA, FVMASignalsShortCross, FVMASignalsLongCross)
    FVMASignalsShortFinal = np.where(inverseFVMA, FVMASignalsLongCross, FVMASignalsShortCross)
    
    # Entry conditions
    long_condition = FVMASignalsLongFinal
    short_condition = FVMASignalsShortFinal
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(long_condition.iloc[i]) and pd.isna(short_condition.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price_guess = close.iloc[i]
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1
        
        if short_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price_guess = close.iloc[i]
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1
    
    return entries