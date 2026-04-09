import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Hardcoded parameters from Pine Script
    donchLength = 20
    minBodyPercentage = 70
    previousBarsCount = 100
    searchFactor = 1.3
    maLength = 20
    maLengthB = 8
    maReaction = 1
    maReactionB = 1
    bullishTrendCondition = 'DIRECCION MEDIA RAPIDA ALCISTA'
    bearishTrendCondition = 'DIRECCION MEDIA RAPIDA BAJISTA'
    filterType = 'CON FILTRADO DE TENDENCIA'
    activateGreenElephantCandles = True
    activateRedElephantCandles = True
    
    high = df['high']
    low = df['low']
    close = df['close']
    open_ = df['open']
    
    # Calculate Wilder ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/previousBarsCount, adjust=False).mean()
    
    #