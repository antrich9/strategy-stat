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
    
    # Default parameters from Pine Script inputs
    i_htfLen = 20
    i_ltfLen = 10
    i_sweepLen = 5
    i_bosLookback = 30
    i_fibEntry = 0.71
    i_requireFVG = False
    i_useSession = False
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    n = len(df)
    entries = []
    trade_num = 1
    
    # State variables (var)
    bearStage = 0
    bullStage = 0
    bearSweepHi = np.nan
    bullSweepLo = np.nan
    bearBosRef = np.nan
    bullBosRef = np.nan
    bearEntry = np.nan
    bullEntry = np.nan
    fvgBearHi = np.nan
    fvgBearLo = np.nan
    fvgBullHi = np.nan
    fvgBullLo = np.nan
    fvgBearBI = np.nan
    fvgBullBI = np.nan
    
    # HTF bias calculation (simulated from same timeframe for backtesting)
    # Using rolling highest/lowest as proxy for HTF swing detection
    htf_swHi = high.rolling(i_htfLen).max()
    htf_swLo = low.rolling(i_htfLen).min()
    htf_mid = (htf_swHi + htf_swLo) / 2.0
    htf_c = close
    
    htfBullBias = htf_c < htf_mid
    htfBearBias = htf_c > htf_mid
    
    # Session filter (always true since i_useSession defaults to false)
    inSession = True
    
    # Precompute swing highs/lows (pivot-based, confirmed)
    swHi = pd.Series(np.nan, index=df.index)
    swLo = pd.Series(np.nan, index=df.index)
    
    for i in range(i_ltfLen, n - i_ltfLen):
        if i >= i_ltfLen and i < n - i_ltfLen:
            window_hi = high.iloc[i - i_ltfLen:i + i_ltfLen + 1]
            window_lo = low.iloc[i - i_ltfLen:i + i_ltfLen + 1]
            if high.iloc[i] == window_hi.max():
                swHi.iloc[i] = high.iloc[i]
            if low.iloc[i] == window_lo.min():
                swLo.iloc[i] = low.iloc[i]
    
    # Track most-recent confirmed pivot values
    lastPivHi = np.nan
    lastPivLo = np.nan
    lastPivHiBI = np.nan
    lastPivLoBI = np.nan
    
    # Precompute sweep conditions
    prevHiX = high.shift(1).rolling(i_sweepLen).max()
    prevLoX = low.shift(1).rolling(i_sweepLen).min()
    
    bearSweepRaw = (high > prevHiX) & (close < prevHiX)
    bullSweepRaw = (low < prevLoX) & (close > prevLoX)
    
    bearSweep = bearSweepRaw & htfBearBias
    bullSweep = bullSweepRaw & htfBullBias
    
    # Precompute FVG zones
    bearFVGraw = high.shift(2) < low
    bullFVGraw = low.shift(2) > high
    
    # BoS reference levels (rolling highest/lowest for lookback window)
    bos_ref_hi = high.rolling(i_bosLookback).max()
    bos_ref_lo = low.rolling(i_bosLookback).min()
    
    # Iterate through bars
    for i in range(1, n):
        # Update pivot tracking
        if not pd.isna(swHi.iloc[i]):
            lastPivHi = swHi.iloc[i]
            lastPivHiBI = i - i_ltfLen
        if not pd.isna(swLo.iloc[i]):
            lastPivLo = swLo.iloc[i]
            lastPivLoBI = i - i_ltfLen
        
        # Update FVG zones
        if bearFVGraw.iloc[i]:
            fvgBearHi = low.iloc[i]
            fvgBearLo = high.iloc[i - 2]
            fvgBearBI = i
        if bullFVGraw.iloc[i]:
            fvgBullHi = low.iloc[i - 2]
            fvgBullLo = high.iloc[i]
            fvgBullBI = i
        
        # Reset stages when opposing sweep arrives
        if bullSweep.iloc[i]:
            bearStage = 0
        if bearSweep.iloc[i]:
            bullStage = 0
        
        # ══════════════ BEARISH PIPELINE ══════════════
        # Stage 0 -> 1: sweep detected
        if bearSweep.iloc[i] and bearStage == 0:
            bearStage = 1
            bearSweepHi = high.iloc[i]
            bearBosRef = low.iloc[:i + 1].rolling(i_bosLookback).min().iloc[i]
        
        # Stage 1 -> 2: BoS (close below reference low) + bearish FVG
        if bearStage == 1 and close.iloc[i] < bearBosRef:
            hasBearFVG = (not pd.isna(fvgBearBI)) and (fvgBearBI >= (i - i_bosLookback))
            if hasBearFVG or not i_requireFVG:
                bearFibHi = bearSweepHi
                bearFibLo = low.iloc[:i + 1].rolling(i_bosLookback).min().iloc[i]
                
                rawEntry = bearFibHi - (bearFibHi - bearFibLo) * i_fibEntry
                
                fvgOK = not i_requireFVG or (not pd.isna(fvgBearHi) and rawEntry <= fvgBearHi and rawEntry >= fvgBearLo)
                
                if fvgOK:
                    bearEntry = rawEntry
                    bearStage = 2
        
        # Fire short order
        if bearStage == 2 and inSession and not pd.isna(bearEntry):
            entry_price = bearEntry
            ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            bearStage = 0
        
        # ══════════════ BULLISH PIPELINE ══════════════
        # Stage 0 -> 1: sweep detected
        if bullSweep.iloc[i] and bullStage == 0:
            bullStage = 1
            bullSweepLo = low.iloc[i]
            bullBosRef = high.iloc[:i + 1].rolling(i_bosLookback).max().iloc[i]
        
        # Stage 1 -> 2: BoS (close above reference high) + bullish FVG
        if bullStage == 1 and close.iloc[i] > bullBosRef:
            hasBullFVG = (not pd.isna(fvgBullBI)) and (fvgBullBI >= (i - i_bosLookback))
            if hasBullFVG or not i_requireFVG:
                bullFibLo = bullSweepLo
                bullFibHi = high.iloc[:i + 1].rolling(i_bosLookback).max().iloc[i]
                
                rawEntry = bullFibLo + (bullFibHi - bullFibLo) * i_fibEntry
                
                fvgOK = not i_requireFVG or (not pd.isna(fvgBullLo) and rawEntry >= fvgBullLo and rawEntry <= fvgBullHi)
                
                if fvgOK:
                    bullEntry = rawEntry
                    bullStage = 2
        
        # Fire long order
        if bullStage == 2 and inSession and not pd.isna(bullEntry):
            entry_price = bullEntry
            ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            bullStage = 0
    
    return entries