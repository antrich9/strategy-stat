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
    
    # Strategy parameters from Pine Script
    swingLen = 10
    sweepWick = True
    chochBars = 20
    obScanBars = 15
    cooldown = 1
    useDateFilter = False
    
    # State machine constants
    ST_SWEEP = 1
    ST_CHOCH = 2
    ST_OB = 3
    ST_DONE = 4
    
    n = len(df)
    results = []
    trade_num = 1
    
    # Calculate pivots
    pivotH = np.full(n, np.nan)
    pivotL = np.full(n, np.nan)
    
    for i in range(swingLen, n - swingLen):
        is_high = True
        is_low = True
        for j in range(1, swingLen + 1):
            if df['high'].iloc[i - j] >= df['high'].iloc[i]:
                is_high = False
            if df['low'].iloc[i - j] <= df['low'].iloc[i]:
                is_low = False
        for j in range(1, swingLen + 1):
            if df['high'].iloc[i + j] > df['high'].iloc[i]:
                is_high = False
            if df['low'].iloc[i + j] < df['low'].iloc[i]:
                is_low = False
        if is_high:
            pivotH[i] = df['high'].iloc[i]
        if is_low:
            pivotL[i] = df['low'].iloc[i]
    
    # Track swing highs and lows
    swingHighs = []
    swingLows = []
    
    # Track setups
    # Setup structure: [isBear, sweptLevel, sweepBar, chochLevel, state, stateBar, obTop, obBot, obBar]
    setups = []
    
    lastSigBar = -1000
    bar_index = 0
    
    for i in range(swingLen, n):
        bar_index = i
        
        # Update swing arrays
        if not np.isnan(pivotH[i]):
            swingHighs.insert(0, df['high'].iloc[i])
            if len(swingHighs) > 20:
                swingHighs.pop()
        
        if not np.isnan(pivotL[i]):
            swingLows.insert(0, df['low'].iloc[i])
            if len(swingLows) > 20:
                swingLows.pop()
        
        recentHigh = swingHighs[0] if len(swingHighs) > 0 else np.nan
        recentLow = swingLows[0] if len(swingLows) > 0 else np.nan
        
        if np.isnan(recentHigh) or np.isnan(recentLow):
            continue
        
        # Sweep detection
        if sweepWick:
            bslSwept = (df['high'].iloc[i] > recentHigh) and (df['close'].iloc[i] < recentHigh)
            sslSwept = (df['low'].iloc[i] < recentLow) and (df['close'].iloc[i] > recentLow)
        else:
            bslSwept = df['high'].iloc[i] > recentHigh
            sslSwept = df['low'].iloc[i] < recentLow
        
        # Create new setups
        if bslSwept:
            setups.insert(0, {
                'isBear': True,
                'sweptLevel': recentHigh,
                'sweepBar': i,
                'chochLevel': recentLow,
                'state': ST_SWEEP,
                'stateBar': i,
                'obTop': np.nan,
                'obBot': np.nan,
                'obBar': 0
            })
        
        if sslSwept:
            setups.insert(0, {
                'isBear': False,
                'sweptLevel': recentLow,
                'sweepBar': i,
                'chochLevel': recentHigh,
                'state': ST_SWEEP,
                'stateBar': i,
                'obTop': np.nan,
                'obBot': np.nan,
                'obBar': 0
            })
        
        # Limit setups array size
        if len(setups) > 8:
            setups.pop()
        
        buySignal = False
        sellSignal = False
        sigSwept = np.nan
        sigOBTop = np.nan
        sigOBBot = np.nan
        sigOBBar = 0
        
        # Process setups
        for s in setups:
            # State transition: SWEEP -> DONE after chochBars
            if s['state'] == ST_SWEEP and (bar_index - s['stateBar']) > chochBars:
                s['state'] = ST_DONE
            
            if s['state'] == ST_SWEEP:
                # Find OB function
                def findOB(isBearSetup, fromBar):
                    obTop = np.nan
                    obBot = np.nan
                    obIdx = 0
                    for lb in range(1, obScanBars + 1):
                        if lb <= fromBar:
                            idx = bar_index - lb
                            if idx >= 0:
                                if isBearSetup:
                                    if df['close'].iloc[idx] > df['open'].iloc[idx]:
                                        obTop = df['high'].iloc[idx]
                                        obBot = min(df['open'].iloc[idx], df['close'].iloc[idx])
                                        obIdx = idx
                                        break
                                else:
                                    if df['close'].iloc[idx] < df['open'].iloc[idx]:
                                        obTop = max(df['open'].iloc[idx], df['close'].iloc[idx])
                                        obBot = df['low'].iloc[idx]
                                        obIdx = idx
                                        break
                    return obTop, obBot, obIdx
                
                if s['isBear']:
                    if df['close'].iloc[i] < s['chochLevel']:
                        oT, oB, oBI = findOB(True, bar_index)
                        if not np.isnan(oT):
                            s['state'] = ST_DONE
                            if (bar_index - lastSigBar) >= cooldown:
                                sellSignal = True
                                sigSwept = s['sweptLevel']
                                sigOBTop = oT
                                sigOBBot = oB
                                sigOBBar = oBI
                else:
                    if df['close'].iloc[i] > s['chochLevel']:
                        oT, oB, oBI = findOB(False, bar_index)
                        if not np.isnan(oT):
                            s['state'] = ST_DONE
                            if (bar_index - lastSigBar) >= cooldown:
                                buySignal = True
                                sigSwept = s['sweptLevel']
                                sigOBTop = oT
                                sigOBBot = oB
                                sigOBBar = oBI
        
        # Execute entries
        if sellSignal:
            entry_price = df['close'].iloc[i]
            lastSigBar = bar_index
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            
            results.append({
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
        
        if buySignal:
            entry_price = df['close'].iloc[i]
            lastSigBar = bar_index
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            
            results.append({
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
    
    return results