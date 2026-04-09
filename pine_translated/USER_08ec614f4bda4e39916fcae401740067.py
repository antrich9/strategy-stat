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
    left = 20
    right = 15
    nPiv = 5
    atrLen = 14
    zoneMult = 0.5
    zonePerc = 3.0
    vwapMinHits = 2
    vfiLen = 130
    vfiCoef = 0.2
    vfiVcoef = 2.5
    vfiSigLen = 5
    vfiConsec = 1
    fkLookback = 3
    enableLong = True
    enableShort = True
    useVwap = True
    useVfi = True
    
    open_s = df['open'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values
    timestamps = df['time'].values
    
    n = len(df)
    
    # ATR (Wilder)
    tr = np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])])
    atr = np.zeros(n)
    atr[atrLen] = tr[:atrLen].mean()
    multiplier = 1.0 - 1.0 / atrLen
    for i in range(atrLen + 1, n):
        atr[i] = atr[i-1] * multiplier + tr[i-1 - atrLen + 1]
    
    # Multi-anchor VWAP
    timestamps_dt = pd.to_datetime(timestamps, unit='s', utc=True)
    day_change = pd.Series(timestamps_dt).dt.date.ne(pd.Series(timestamps_dt).dt.date.shift(1)).values
    week_change = pd.Series(timestamps_dt).dt.to_period('W').ne(pd.Series(timestamps_dt).dt.to_period('W').shift(1)).values
    month_change = pd.Series(timestamps_dt).dt.to_period('M').ne(pd.Series(timestamps_dt).dt.to_period('M').shift(1)).values
    
    hl2 = (high + low) / 2.0
    
    vwapS = np.zeros(n)
    vwapW = np.zeros(n)
    vwapM = np.zeros(n)
    
    pvSum_S = 0.0
    volSum_S = 0.0
    pvSum_W = 0.0
    volSum_W = 0.0
    pvSum_M = 0.0
    volSum_M = 0.0
    
    for i in range(n):
        if day_change[i] or i == 0:
            pvSum_S = hl2[i] * volume[i]
            volSum_S = volume[i]
        else:
            pvSum_S += hl2[i] * volume[i]
            volSum_S += volume[i]
        vwapS[i] = pvSum_S / volSum_S if volSum_S > 0 else 0
        
        if week_change[i] or i == 0:
            pvSum_W = hl2[i] * volume[i]
            volSum_W = volume[i]
        else:
            pvSum_W += hl2[i] * volume[i]
            volSum_W += volume[i]
        vwapW[i] = pvSum_W / volSum_W if volSum_W > 0 else 0
        
        if month_change[i] or i == 0:
            pvSum_M = hl2[i] * volume[i]
            volSum_M = volume[i]
        else:
            pvSum_M += hl2[i] * volume[i]
            volSum_M += volume[i]
        vwapM[i] = pvSum_M / volSum_M if volSum_M > 0 else 0
    
    aboveCount = ((close > vwapS).astype(int) + (close > vwapW).astype(int) + (close > vwapM).astype(int))
    belowCount = ((close < vwapS).astype(int) + (close < vwapW).astype(int) + (close < vwapM).astype(int))
    vwapBullish = aboveCount >= vwapMinHits
    vwapBearish = belowCount >= vwapMinHits
    
    # VFI (LazyBear)
    typical = (high + low + close) / 3.0
    inter = np.zeros(n)
    inter[1:] = np.log(typical[1:]) - np.log(typical[:-1])
    
    vinter = np.zeros(n)
    for i in range(30, n):
        vinter[i] = np.std(inter[i-30:i])
    
    cutoff = vfiCoef * vinter * close
    
    vave = np.zeros(n)
    for i in range(vfiLen, n):
        vave[i] = np.mean(volume[i-vfiLen:i])
    vave_prev = np.roll(vave, 1)
    vave_prev[0] = vave[vfiLen] if vfiLen < n else 1.0
    
    vmax = vave_prev * vfiVcoef
    vc = np.minimum(volume, vmax)
    
    mf = np.zeros(n)
    mf[1:] = typical[1:] - typical[:-1]
    
    vcp = np.zeros(n)
    vcp = np.where(mf > cutoff, vc, vcp)
    vcp = np.where(mf < -cutoff, -vc, vcp)
    
    vfi = np.zeros(n)
    for i in range(vfiLen, n):
        vcp_sum = np.sum(vcp[i-vfiLen+1:i+1])
        vfi_raw = vcp_sum / vave_prev[i] if vave_prev[i] != 0 else 0
        vfi[i] = np.mean(vfi_raw)
    vfi = pd.Series(vfi).ewm(span=3, adjust=False).mean().values
    
    vfima = pd.Series(vfi).ewm(span=vfiSigLen, adjust=False).mean().values
    
    vfiBuy = np.ones(n, dtype=bool)
    vfiSell = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(vfiConsec):
            idx = i - j
            if idx >= 0:
                if vfi[idx] <= vfima[idx]:
                    vfiBuy[i] = False
                if vfi[idx] >= vfima[idx]:
                    vfiSell[i] = False
    
    # Heikin Ashi
    haClose = (open_s + high + low + close) / 4.0
    haOpen = np.zeros(n)
    haOpen[0] = (open_s[0] + close[0]) / 2.0
    for i in range(1, n):
        haOpen[i] = (haOpen[i-1] + haClose[i-1]) / 2.0 if not np.isnan(haOpen[i-1]) else (open_s[i] + close[i]) / 2.0
    
    srcHigh = np.maximum(haClose, haOpen)
    srcLow = np.minimum(haClose, haOpen)
    
    perc = close * (zonePerc / 100)
    
    # Calculate halfBand at right-shift bars
    halfBand = np.zeros(n)
    for i in range(right, n):
        atr_val = atr[i]
        zone_width = min(atr_val * zoneMult, perc[i])
        halfBand[i] = zone_width / 2.0
    
    # Pivot detection
    pivHigh = np.full(n, np.nan)
    pivLow = np.full(n, np.nan)
    
    for i in range(left + right, n):
        max_high = srcHigh[i-left:i+1]
        if np.argmax(max_high) == left and srcHigh[i] == np.max(srcHigh[i-left:i+1]):
            pivHigh[i] = srcHigh[i]
        
        min_low = srcLow[i-left:i+1]
        if np.argmin(min_low) == left and srcLow[i] == np.min(srcLow[i-left:i+1]):
            pivLow[i] = srcLow[i]
    
    # Build resistance and support zones
    resTop = []
    resBtm = []
    supTop = []
    supBtm = []
    
    for i in range(n):
        if not np.isnan(pivHigh[i]):
            hb = halfBand[i] if i < len(halfBand) else halfBand[-1]
            resTop.insert(0, pivHigh[i] + hb)
            resBtm.insert(0, pivHigh[i] - hb)
            if len(resTop) > nPiv:
                resTop.pop()
                resBtm.pop()
        
        if not np.isnan(pivLow[i]):
            hb = halfBand[i] if i < len(halfBand) else halfBand[-1]
            supTop.insert(0, pivLow[i] + hb)
            supBtm.insert(0, pivLow[i] - hb)
            if len(supTop) > nPiv:
                supTop.pop()
                supBtm.pop()
    
    # Fakeout detection
    fakeShort = np.zeros(n, dtype=bool)
    fakeLong = np.zeros(n, dtype=bool)
    fkShortHigh_val = np.full(n, np.nan)
    fkLongLow_val = np.full(n, np.nan)
    
    for i in range(n):
        if len(resTop) > 0 and len(resBtm) > 0:
            piercedAbove = False
            for j in range(1, fkLookback + 1):
                if i - j >= 0:
                    if high[i - j] > resTop[0]:
                        piercedAbove = True
                        break
            rejectedBack = close[i] < resBtm[0] and close[i] < open_s[i]
            if piercedAbove and rejectedBack:
                fakeShort[i] = True
                fkShortHigh_val[i] = resTop[0]
        
        if len(supTop) > 0 and len(supBtm) > 0:
            piercedBelow = False
            for j in range(1, fkLookback + 1):
                if i - j >= 0:
                    if low[i - j] < supBtm[0]:
                        piercedBelow = True
                        break
            reclaimedBack = close[i] > supTop[0] and close[i] > open_s[i]
            if piercedBelow and reclaimedBack:
                fakeLong[i] = True
                fkLongLow_val[i] = supBtm[0]
    
    # Entry signals
    vwapOkShort = vwapBearish if useVwap else np.ones(n, dtype=bool)
    vfiOkShort = vfiSell if useVfi else np.ones(n, dtype=bool)
    shortSignal = fakeShort & vwapOkShort & vfiOkShort & enableShort
    
    vwapOkLong = vwapBullish if useVwap else np.ones(n, dtype=bool)
    vfiOkLong = vfiBuy if useVfi else np.ones(n, dtype=bool)
    longSignal = fakeLong & vwapOkLong & vfiOkLong & enableLong
    
    entries = []
    trade_num = 1
    
    for i in range(n):
        if shortSignal[i] and np.isfinite(close[i]):
            entry_price = float(close[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(timestamps[i]),
                'entry_time': datetime.fromtimestamp(timestamps[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif longSignal[i] and np.isfinite(close[i]):
            entry_price = float(close[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(timestamps[i]),
                'entry_time': datetime.fromtimestamp(timestamps[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries