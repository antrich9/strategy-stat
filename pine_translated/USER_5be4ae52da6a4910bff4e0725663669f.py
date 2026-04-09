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
    
    # Parameters
    swingLength = 10
    fibLevel = 0.71
    barsAfterBOS = 5
    
    # Initialize
    bias = "None"
    bulltap = 0
    beartap = 0
    lastH = np.nan
    lastL = np.nan
    fibHigh = np.nan
    fibLow = np.nan
    bosBarIndex = np.nan
    isBearBOS = False
    isBullBOS = False
    prevBosBarIndex = np.nan
    bosLegHasFVG = False
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        # Current values
        close = df['close'].iloc[i]
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        
        # Previous values
        if i > 0:
            close_prev = df['close'].iloc[i-1]
            high_prev = df['high'].iloc[i-1]
            low_prev = df['low'].iloc[i-1]
        else:
            close_prev = close
            high_prev = high
            low_prev = low
        
        # Two bars back for FVG
        if i >= 2:
            low_2 = df['low'].iloc[i-2]
            high_2 = df['high'].iloc[i-2]
        else:
            low_2 = low
            high_2 = high
        
        # Swing detection
        # Using simple pivot detection (can be optimized)
        if i >= swingLength:
            # Check for pivot high
            isPivotHigh = True
            for j in range(1, swingLength + 1):
                if df['high'].iloc[i - swingLength + j] > high:
                    isPivotHigh = False
                    break
            if isPivotHigh and df['high'].iloc[i - swingLength] == high:
                lastH = high
                
            # Check for pivot low
            isPivotLow = True
            for j in range(1, swingLength + 1):
                if df['low'].iloc[i - swingLength + j] < low:
                    isPivotLow = False
                    break
            if isPivotLow and df['low'].iloc[i - swingLength] == low:
                lastL = low
        
        # Update pivot values when swing detection occurs
        if not np.isnan(lastH):
            lastH_val = lastH
        else:
            lastH_val = np.nan
            
        if not np.isnan(lastL):
            lastL_val = lastL
        else:
            lastL_val = np.nan
        
        # BOS Logic
        if close < lastL_val and bias != "Bearish":
            bias = "Bearish"
            fibHigh = lastH_val
            fibLow = lastL_val
            bulltap = 0
            beartap = 0
            bosBarIndex = i
            isBearBOS = True
            isBullBOS = False
        
        if close > lastH_val and bias != "Bullish":
            bias = "Bullish"
            fibHigh = lastH_val
            fibLow = lastL_val
            bulltap = 0
            beartap = 0
            bosBarIndex = i
            isBearBOS = False
            isBullBOS = True
        
        # Update fib legs without new BOS
        if bias == "Bearish" and not np.isnan(lastH_val) and lastH_val < fibHigh:
            fibHigh = lastH_val
            bulltap = 0
            beartap = 0
        
        if bias == "Bearish" and not np.isnan(lastL_val):
            fibLow = lastL_val
            bulltap = 0
            beartap = 0
        
        if bias == "Bullish" and not np.isnan(lastL_val) and lastL_val > fibLow:
            fibLow = lastL_val
            bulltap = 0
            beartap = 0
        
        if bias == "Bullish" and not np.isnan(lastH_val):
            fibHigh = lastH_val
            bulltap = 0
            beartap = 0
        
        # FVG Detection
        bearFVG = low_2 > high
        bullFVG = low < high_2
        
        # BOS Leg FVG tracking
        newBOS = bosBarIndex != prevBosBarIndex
        prevBosBarIndex = bosBarIndex
        
        if newBOS:
            bosLegHasFVG = False
        
        onBOSLeg = not np.isnan(bosBarIndex) and i >= bosBarIndex and i <= bosBarIndex + barsAfterBOS
        
        if onBOSLeg and not np.isnan(fibHigh) and not np.isnan(fibLow):
            if isBearBOS and bearFVG and high >= fibLow and low_2 <= fibHigh:
                bosLegHasFVG = True
            if isBullBOS and bullFVG and high_2 >= fibLow and low <= fibHigh:
                bosLegHasFVG = True
        
        # Fibonacci levels
        fib071 = np.nan
        fib100 = np.nan
        fib000 = np.nan
        
        if not np.isnan(fibHigh) and not np.isnan(fibLow):
            if bias == "Bearish":
                fib071 = fibLow + (fibHigh - fibLow) * fibLevel
                fib100 = fibHigh
                fib000 = fibLow
            elif bias == "Bullish":
                fib071 = fibHigh - (fibHigh - fibLow) * fibLevel
                fib100 = fibLow
                fib000 = fibHigh
        
        # Entry zone
        inZone = False
        if not np.isnan(fib071) and not np.isnan(fibHigh) and not np.isnan(fibLow):
            zone_width = abs(fibHigh - fibLow) * 0.15
            inZone = close >= (fib071 - zone_width) and close <= (fib071 + zone_width)
        
        # Entry flags
        bullEntry = bias == "Bullish" and bosLegHasFVG and inZone
        bearEntry = bias == "Bearish" and bosLegHasFVG and inZone
        
        # HTF Filter (simplified - would need actual HTF data)
        # For this conversion, we'll assume HTF conditions are always true
        # or we can calculate EMA on daily data if available
        bullHTF = True
        bearHTF = True
        
        # Tap counting (only on confirmed bars - simplified)
        if i > 0:
            if low < fib071 and low_prev >= fib071 and bullEntry:
                bulltap += 1
            
            if high > fib071 and high_prev <= fib071 and bearEntry:
                beartap += 1
        
        # Entry execution
        if bulltap == 1 and bullEntry and bullHTF:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': fib071 if not np.isnan(fib071) else close,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': fib071 if not np.isnan(fib071) else close,
                'raw_price_b': fib071 if not np.isnan(fib071) else close
            })
            trade_num += 1
            bulltap = 0
        
        if beartap == 1 and bearEntry and bearHTF:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': fib071 if not np.isnan(fib071) else close,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': fib071 if not np.isnan(fib071) else close,
                'raw_price_b': fib071 if not np.isnan(fib071) else close
            })
            trade_num += 1
            beartap = 0
    
    return entries

# Note: This implementation simplifies some aspects:
# 1. HTF filter is assumed to be true (would need actual HTF data)
# 2. Pivot detection uses a simple approach
# 3. Bar state is confirmed is simplified