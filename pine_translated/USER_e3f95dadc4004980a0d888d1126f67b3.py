import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Parameters from script
    pivotLen = 14
    fvgWaitBars = 10
    fvgMinTicks = 3
    waitForFVG = True
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # ATR (Wilder smoothing)
    tr = pd.concat([high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # Bullish FVG: low > high[2]
    bullishFVG = low > high.shift(2)
    bullishFVGSize = np.where(bullishFVG, low - high.shift(2), 0)
    bullishFVGValid = bullishFVG & (bullishFVGSize >= fvgMinTicks * atr)
    
    # Bearish FVG: high < low[2]
    bearishFVG = high < low.shift(2)
    bearishFVGSize = np.where(bearishFVG, low.shift(2) - high, 0)
    bearishFVGValid = bearishFVG & (bearishFVGSize >= fvgMinTicks * atr)
    
    # Pivot detection (shift by pivotLen to align with pivot bar)
    pivothigh = high.shift(pivotLen)
    pivotlow = low.shift(pivotLen)
    
    lastBearSweepBar = None
    lastBearSweepPrice = np.nan
    lastBullSweepBar = None
    lastBullSweepPrice = np.nan
    
    trade_num = 0
    entries = []
    
    for i in range(pivotLen, len(df)):
        ts = df['time'].iloc[i]
        
        # Bear zone (pivot high)
        if pivothigh.iloc[i] == high.iloc[i-pivotLen:pivotLen*2+1].max():
            lastBearSweepBar = i
            lastBearSweepPrice = high.iloc[i-pivotLen]
        
        # Bull zone (pivot low)
        if pivotlow.iloc[i] == low.iloc[i-pivotLen:pivotLen*2+1].min():
            lastBullSweepBar = i
            lastBullSweepPrice = low.iloc[i-pivotLen]
        
        # Sweep detection
        if i > 0 and not pd.isna(lastBearSweepPrice):
            if high.iloc[i] > lastBearSweepPrice and close.iloc[i] < lastBearSweepPrice:
                lastBearSweepBar = i
                lastBearSweepPrice = high.iloc[i]
        
        if i > 0 and not pd.isna(lastBullSweepPrice):
            if low.iloc[i] < lastBullSweepPrice and close.iloc[i] > lastBullSweepPrice:
                lastBullSweepBar = i
                lastBullSweepPrice = low.iloc[i]
        
        # Calculate bars since sweep
        barsSinceBearSweep = i - lastBearSweepBar if lastBearSweepBar is not None else 999
        barsSinceBullSweep = i - lastBullSweepBar if lastBullSweepBar is not None else 999
        
        # Clear old sweeps
        if barsSinceBearSweep > fvgWaitBars:
            lastBearSweepBar = None
            lastBearSweepPrice = np.nan
        if barsSinceBullSweep > fvgWaitBars:
            lastBullSweepBar = None
            lastBullSweepPrice = np.nan
        
        # Short: Bear sweep (high taken) + Bearish FVG
        shortSetup = (barsSinceBearSweep > 0 and barsSinceBearSweep <= fvgWaitBars)
        shortEntry = shortSetup and (bearishFVGValid.iloc[i] if waitForFVG else True)
        
        # Long: Bull sweep (low taken) + Bullish FVG
        longSetup = (barsSinceBullSweep > 0 and barsSinceBullSweep <= fvgWaitBars)
        longEntry = longSetup and (bullishFVGValid.iloc[i] if waitForFVG else True)
        
        if shortEntry:
            trade_num += 1
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            lastBearSweepBar = None
            lastBearSweepPrice = np.nan
        
        if longEntry:
            trade_num += 1
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            lastBullSweepBar = None
            lastBullSweepPrice = np.nan
    
    return entries