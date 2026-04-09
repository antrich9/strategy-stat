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
    
    # Parameters from Pine Script
    alphaLength = 20
    gammaLength = 20
    per_tbi = 14
    per2_tbi = 14
    length50line = 50
    length50lineCondition = 'Above'
    highlightMovementsTBI = True
    mmaLengthTDFI = 13
    smmaLengthTDFI = 13
    nLengthTDFI = 3
    lookbackTDFI = 13
    filterHighTDFI = 0.05
    filterLowTDFI = -0.05
    useTBI = True
    crossTBI = True
    inverseTBI = False
    useTDFI = True
    crossTDFI = True
    inverseTDFI = True
    
    close = df['close']
    low = df['low']
    high = df['high']
    
    # Calculate HEMA iteratively
    alpha = 2 / (alphaLength + 1)
    gamma = 2 / (gammaLength + 1)
    
    hema = np.zeros(len(df))
    b = np.zeros(len(df))
    
    for i in range(1, len(df)):
        src = close.iloc[i]
        hema_prev = hema[i-1] if not np.isnan(hema[i-1]) else src
        b_prev = b[i-1] if not np.isnan(b[i-1]) else 0
        
        hema[i] = (1 - alpha) * (hema_prev + b_prev) + alpha * src
        b[i] = (1 - gamma) * b_prev + gamma * (hema[i] - hema_prev)
    
    hema_series = pd.Series(hema, index=df.index)
    
    # HEMA color: green if hema > hema[1], red otherwise
    hema_green = hema_series > hema_series.shift(1)
    hema_red = hema_series < hema_series.shift(1)
    
    # TBI calculations
    lowest_low_1 = low.shift(1).rolling(window=per_tbi).min()
    lowest_low_per = low.rolling(window=per_tbi).min()
    loc_tbi = (low < lowest_low_1) & (low <= lowest_low_per)
    
    highest_high_1 = high.shift(1).rolling(window=per2_tbi).max()
    highest_high_per = high.rolling(window=per2_tbi).max()
    loc2_tbi = (high > highest_high_1) & (high >= highest_high_per)
    
    bottom_tbi = np.zeros(len(df))
    top_tbi = np.zeros(len(df))
    
    for i in range(len(df)):
        if loc_tbi.iloc[i]:
            bottom_tbi[i] = 0
        else:
            bottom_tbi[i] = bottom_tbi[i-1] + 1 if i > 0 else np.nan
            
        if loc2_tbi.iloc[i]:
            top_tbi[i] = 0
        else:
            top_tbi[i] = top_tbi[i-1] + 1 if i > 0 else np.nan
    
    bottom_tbi_series = pd.Series(bottom_tbi, index=df.index)
    top_tbi_series = pd.Series(top_tbi, index=df.index)
    
    # TBI conditions
    condtion50Long = bottom_tbi_series > length50line if length50lineCondition == 'Above' else bottom_tbi_series < length50line
    condtion50Short = top_tbi_series > length50line if length50lineCondition == 'Above' else top_tbi_series < length50line
    
    basicLongConditionTBI = (top_tbi_series < bottom_tbi_series) & condtion50Long
    basicShortConditionTBI = (top_tbi_series > bottom_tbi_series) & condtion50Short
    
    TBISignalsLong = basicLongConditionTBI if highlightMovementsTBI else (top_tbi_series < bottom_tbi_series)
    TBISignalsShort = basicShortConditionTBI if highlightMovementsTBI else (top_tbi_series > bottom_tbi_series)
    
    TBISignalsLong = pd.Series(TBISignalsLong, index=df.index)
    TBISignalsShort = pd.Series(TBISignalsShort, index=df.index)
    
    TBISignalsLongCross = (~TBISignalsLong.shift(1).fillna(False)) & TBISignalsLong if crossTBI else TBISignalsLong
    TBISignalsShortCross = (~TBISignalsShort.shift(1).fillna(False)) & TBISignalsShort if crossTBI else TBISignalsShort
    
    TBISignalsLongFinal = TBISignalsShortCross if (useTBI and inverseTBI) else TBISignalsLongCross
    TBISignalsShortFinal = TBISignalsLongCross if (useTBI and inverseTBI) else TBISignalsShortCross
    
    # TDFI calculations (using default EMA mode)
    price_scaled = close * 1000
    
    mmaTDFI = price_scaled.ewm(span=mmaLengthTDFI, adjust=False).mean()
    smmaTDFI = mmaTDFI.ewm(span=smmaLengthTDFI, adjust=False).mean()
    
    impetmmaTDFI = mmaTDFI - mmaTDFI.shift(1)
    impetsmmaTDFI = smmaTDFI - smmaTDFI.shift(1)
    divmaTDFI = np.abs(mmaTDFI - smmaTDFI)
    averimpetTDFI = (impetmmaTDFI + impetsmmaTDFI) / 2
    
    tdfTDFI = divmaTDFI * np.power(averimpetTDFI, nLengthTDFI)
    
    lookback_period = lookbackTDFI * nLengthTDFI
    highest_tdf = tdfTDFI.rolling(window=lookback_period).max()
    
    signalTDFI = tdfTDFI / highest_tdf
    
    signalLongTDFI = (signalTDFI > filterHighTDFI)
    signalShortTDFI = (signalTDFI < filterLowTDFI)
    
    finalLongSignalTDFI = signalShortTDFI if inverseTDFI else signalLongTDFI
    finalShortSignalTDFI = signalLongTDFI if inverseTDFI else signalShortTDFI
    
    # Final entry conditions
    longCondition = hema_green & (signalTDFI > 0) & TBISignalsLongFinal
    shortCondition = hema_red & (signalTDFI < 0) & TBISignalsShortFinal
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if pd.isna(signalTDFI.iloc[i]) or pd.isna(hema_series.iloc[i]):
            continue
            
        direction = None
        if longCondition.iloc[i]:
            direction = 'long'
        elif shortCondition.iloc[i]:
            direction = 'short'
        
        if direction:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
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