import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    
    pip = 0.0002
    offset = pip * 2
    
    ema8 = df['close'].ewm(span=8, adjust=False).mean()
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    
    body = (df['close'] - df['open']).abs()
    rng = df['high'] - df['low']
    isDoji = (rng > 0) & (body / rng <= 0.30)
    
    emaBull = (ema8 > ema20) & (ema20 > ema50)
    emaBear = (ema8 < ema20) & (ema20 < ema50)
    
    bodyInUpper33 = (df['open'] >= df['low'] + rng * 0.67) & (df['close'] >= df['low'] + rng * 0.67)
    bodyInLower33 = (df['open'] <= df['low'] + rng * 0.33) & (df['close'] <= df['low'] + rng * 0.33)
    closeInUpper33 = (df['close'] - df['low']) / rng >= 0.67
    closeInLower33 = (df['high'] - df['close']) / rng >= 0.67
    
    bullSignal = emaBull & isDoji & (df['low'] <= ema8) & closeInUpper33 & bodyInUpper33
    bearSignal = emaBear & isDoji & (df['high'] >= ema8) & closeInLower33 & bodyInLower33
    
    nyHour = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).hour)
    inAsianSession = nyHour >= 19
    
    asianStarted = inAsianSession & (~inAsianSession).shift(1).fillna(False)
    asianEnded = (~inAsianSession) & inAsianSession.shift(1).fillna(False)
    
    asianHigh = df['high'].cummax().where(asianStarted, df['high'])
    asianLow = df['low'].cummin().where(asianStarted, df['low'])
    
    asianSweptHigh = (df['high'] > asianHigh.shift(1)).fillna(False)
    asianSweptLow = (df['low'] < asianLow.shift(1)).fillna(False)
    
    newDay = df['time'].diff() >= 86400000
    
    tmpPH = df['high'].cummax()
    tmpPL = df['low'].cummin()
    pdHigh = tmpPH.shift(1).where(newDay.shift(1).fillna(False), pdHigh.shift(1))
    pdLow = tmpPL.shift(1).where(newDay.shift(1).fillna(False), pdLow.shift(1))
    
    for i in range(1, len(df)):
        if newDay.iloc[i]:
            pdHigh.iloc[i] = tmpPH.iloc[i-1]
            pdLow.iloc[i] = tmpPL.iloc[i-1]
    
    pdSweptHigh = (df['high'] > pdHigh.shift(1)).fillna(False)
    pdSweptLow = (df['low'] < pdLow.shift(1)).fillna(False)
    
    bullishBias = (df['close'] > pdHigh.shift(1)) | ((df['low'] < pdLow.shift(1)) & (df['close'] > pdLow.shift(1)))
    bearishBias = (df['close'] < pdLow.shift(1)) | ((df['high'] > pdHigh.shift(1)) & (df['close'] < pdHigh.shift(1)))
    
    asianLongOk = asianSweptLow & ~asianSweptHigh
    asianShortOk = asianSweptHigh & ~asianSweptLow
    pdLongOk = pdSweptLow & ~pdSweptHigh
    pdShortOk = pdSweptHigh & ~pdSweptLow
    
    longSweepOk = asianLongOk & pdLongOk & bullishBias
    shortSweepOk = asianShortOk & pdShortOk & bearishBias
    
    gmtHour = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).hour)
    inLondon = (gmtHour >= 7) & (gmtHour < 10)
    inNYAM = (gmtHour >= 13) & (gmtHour < 16)
    in_time_window = inLondon | inNYAM
    
    pendLongPrice = bullSignal.shift(1).where(bullSignal.shift(1), np.nan)
    pendShortPrice = bearSignal.shift(1).where(bearSignal.shift(1), np.nan)
    pendLongTrigger = bullSignal.shift(1) & longSweepOk.shift(1) & in_time_window.shift(1)
    pendShortTrigger = bearSignal.shift(1) & shortSweepOk.shift(1) & in_time_window.shift(1)
    
    entries = []
    trade_num = 1
    
    for i in range(2, len(df)):
        if pd.isna(ema8.iloc[i]) or pd.isna(ema20.iloc[i]) or pd.isna(ema50.iloc[i]):
            continue
        
        if bullSignal.iloc[i-1] and longSweepOk.iloc[i-1] and in_time_window.iloc[i-1]:
            pendLongPrice.iloc[i] = df['high'].iloc[i-1] + offset
        if bearSignal.iloc[i-1] and shortSweepOk.iloc[i-1] and in_time_window.iloc[i-1]:
            pendShortPrice.iloc[i] = df['low'].iloc[i-1] - offset
        
        if pd.isna(pendLongPrice.iloc[i-1]) and not pd.isna(pendLongPrice.iloc[i]):
            pendLongPrice.iloc[i] = pendLongPrice.iloc[i]
        elif not pd.isna(pendLongPrice.iloc[i-1]):
            pendLongPrice.iloc[i] = pendLongPrice.iloc[i-1]
        
        if pd.isna(pendShortPrice.iloc[i-1]) and not pd.isna(pendShortPrice.iloc[i]):
            pendShortPrice.iloc[i] = pendShortPrice.iloc[i]
        elif not pd.isna(pendShortPrice.iloc[i-1]):
            pendShortPrice.iloc[i] = pendShortPrice.iloc[i-1]
    
    for i in range(1, len(df)):
        if pd.isna(ema8.iloc[i]) or pd.isna(ema20.iloc[i]) or pd.isna(ema50.iloc[i]):
            continue
        
        if i > 0:
            if bullSignal.iloc[i-1] and longSweepOk.iloc[i-1] and in_time_window.iloc[i-1]:
                pendLongPrice = df['high'].iloc[i-1] + offset
            else:
                pendLongPrice = np.nan
            if bearSignal.iloc[i-1] and shortSweepOk.iloc[i-1] and in_time_window.iloc[i-1]:
                pendShortPrice = df['low'].iloc[i-1] - offset
            else:
                pendShortPrice = np.nan