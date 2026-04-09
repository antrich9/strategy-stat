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
    # Make a copy to avoid modifying original
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('time')
    
    # Parameters
    tickTol = 8
    atrLen = 10
    atrMult = 0.8
    mssLen = 7
    confirmType = "ATR+MSS"
    useFVG = True
    fvgLookback = 10
    fvgMinSizeTicks = 3
    minBodyPct = 0.32
    maxTopWickPct = 0.80
    maxBotWickPct = 0.80
    useVolSpike = False
    volLen = 20
    volMult = 1.4
    useExtremeWick = False
    extremeWickCut = 0.90
    usePD = True
    usePW = True
    htfTF1 = "240"
    htfPivotLen1 = 5
    srBlockTicks1 = 30
    useHTF1 = True
    htfTF2 = "D"
    htfPivotLen2 = 5
    srBlockTicks2 = 30
    useHTF2 = True
    
    # Estimate mintick from data (smallest price increment)
    mintick = df['high'].diff().dropna()
    mintick = mintick[mintick > 0]
    if len(mintick) > 0:
        mintick = mintick.min()
    else:
        mintick = 0.01
    tickSize = mintick * tickTol
    
    # Wilder ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/atrLen, adjust=False).mean()
    
    # Wilder RSI helper
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # HTF levels calculation
    # Previous Day High/Low - shift by 1 day
    df['date'] = df.index.date
    daily_agg = df.groupby('date').agg({'high': 'max', 'low': 'min'})
    daily_agg['pdh'] = daily_agg['high'].shift(1)
    daily_agg['pdl'] = daily_agg['low'].shift(1)
    df['pdh'] = df['date'].map(daily_agg['pdh'])
    df['pdl'] = df['date'].map(daily_agg['pdl'])
    
    # Previous Week High/Low - shift by 1 week
    df['week'] = df.index.to_period('W')
    weekly_agg = df.groupby('week').agg({'high': 'max', 'low': 'min'})
    weekly_agg['pwh'] = weekly_agg['high'].shift(1)
    weekly_agg['pwl'] = weekly_agg['low'].shift(1)
    df['pwh'] = df['week'].map(weekly_agg['pwh'])
    df['pwl'] = df['week'].map(weekly_agg['pwl'])
    
    # FVG Detection
    high_2 = df['high'].shift(2)
    low_2 = df['low'].shift(2)
    bullFVG = df['low'] > high_2
    bearFVG = df['high'] < low_2
    bullFVGSize = np.where(bullFVG, df['low'] - high_2, 0.0)
    bearFVGSize = np.where(bearFVG, low_2 - df['high'], 0.0)
    bullFVGValid = bullFVG & (bullFVGSize / mintick >= fvgMinSizeTicks)
    bearFVGValid = bearFVG & (bearFVGSize / mintick >= fvgMinSizeTicks)
    
    # Sweep conditions
    def isBearSweep(level):
        return (df['high'] > level + tickSize) & (df['close'] < level)
    
    def isBullSweep(level):
        return (df['low'] < level - tickSize) & (df['close'] > level)
    
    def closeDistBearOK(level):
        return (level - df['close']) / mintick >= 2
    
    def closeDistBullOK(level):
        return (df['close'] - level) / mintick >= 2
    
    bearSweepPD = usePD & isBearSweep(df['pdh']) & closeDistBearOK(df['pdh'])
    bearSweepPW = usePW & isBearSweep(df['pwh']) & closeDistBearOK(df['pwh'])
    bullSweepPD = usePD & isBullSweep(df['pdl']) & closeDistBullOK(df['pdl'])
    bullSweepPW = usePW & isBullSweep(df['pwl']) & closeDistBullOK(df['pwl'])
    
    # Candle anatomy
    rng = df['high'] - df['low']
    body = np.abs(df['close'] - df['open'])
    topWick = df['high'] - np.maximum(df['open'], df['close'])
    botWick = np.minimum(df['open'], df['close']) - df['low']
    
    bodyPct = np.where(rng > 0, body / rng, 0.0)
    topWickPct = np.where(rng > 0, topWick / rng, 0.0)
    botWickPct = np.where(rng > 0, botWick / rng, 0.0)
    
    bodyOK = bodyPct >= minBodyPct
    wickBearOK = topWickPct <= maxTopWickPct
    wickBullOK = botWickPct <= maxBotWickPct
    
    bearSweep = (bearSweepPD | bearSweepPW) & bodyOK & wickBearOK
    bullSweep = (bullSweepPD | bullSweepPW) & bodyOK & wickBullOK
    
    # Confirmations
    atrConf = atr
    rangeRatio = np.where(atrConf > 0, (df['high'] - df['low']) / atrConf, 0.0)
    atrConfirmBear = (df['close'] < df['open']) & (rangeRatio > atrMult)
    atrConfirmBull = (df['close'] > df['open']) & (rangeRatio > atrMult)
    
    lowest_mss = df['low'].rolling(mssLen).min().shift(1)
    highest_mss = df['high'].rolling(mssLen).max().shift(1)
    mssConfirmBear = df['close'] < lowest_mss
    mssConfirmBull = df['close'] > highest_mss
    
    def getBearConfirm():
        if confirmType == "None":
            return pd.Series(True, index=df.index)
        elif confirmType == "ATR":
            return atrConfirmBear
        elif confirmType == "MSS":
            return mssConfirmBear
        else:  # "ATR+MSS"
            return atrConfirmBear & mssConfirmBear
    
    def getBullConfirm():
        if confirmType == "None":
            return pd.Series(True, index=df.index)
        elif confirmType == "ATR":
            return atrConfirmBull
        elif confirmType == "MSS":
            return mssConfirmBull
        else:  # "ATR+MSS"
            return atrConfirmBull & mssConfirmBull
    
    baseBear = bearSweep & getBearConfirm()
    baseBull = bullSweep & getBullConfirm()
    
    # Track sweep bars using rolling window approach
    # For each bar, find how many bars since last sweep
    bar_index = np.arange(len(df))
    
    # Find indices where sweeps occurred
    bear_sweep_indices = bar_index[baseBear.values]
    bull_sweep_indices = bar_index[baseBull.values]
    
    # For each bar, calculate bars since last sweep
    barsFromBearSweep = pd.Series(len(df), index=df.index)
    barsFromBullSweep = pd.Series(len(df), index=df.index)
    
    for i in range(len(df)):
        bear_bars = i - bear_sweep_indices[bear_sweep_indices < i]
        bull_bars = i - bull_sweep_indices[bull_sweep_indices < i]
        if len(bear_bars) > 0:
            barsFromBearSweep.iloc[i] = bear_bars[-1]
        else:
            barsFromBearSweep.iloc[i] = 999
        if len(bull_bars) > 0:
            barsFromBullSweep.iloc[i] = bull_bars[-1]
        else:
            barsFromBullSweep.iloc[i] = 999
    
    # FVG After Sweep Logic
    longSweepValid = (barsFromBearSweep > 0) & (barsFromBearSweep <= fvgLookback)
    longFVGConfirm = bullFVGValid & longSweepValid
    shortSweepValid = (barsFromBullSweep > 0) & (barsFromBullSweep <= fvgLookback)
    shortFVGConfirm = bearFVGValid & shortSweepValid
    
    # Extra filters
    volSma = df['volume'].rolling(volLen).mean()
    volOK = df['volume'].notna() & volSma.notna() & (df['volume'] > volSma * volMult)
    extremeWickBear = topWickPct > extremeWickCut
    extremeWickBull = botWickPct > extremeWickCut
    
    extraBearPass = ((not useVolSpike) or volOK) & ((not useExtremeWick) or ~extremeWickBear)
    extraBullPass = ((not useVolSpike) or volOK) & ((not useExtremeWick) or ~extremeWickBull)
    
    # Final base conditions with FVG
    if useFVG:
        finalBear_base = extraBearPass & longFVGConfirm
        finalBull_base = extraBullPass & shortFVGConfirm
    else:
        finalBear_base = extraBearPass & (baseBear | (barsFromBearSweep > 0) & (barsFromBearSweep <= fvgLookback))
        finalBull_base = extraBullPass & (baseBull | (barsFromBullSweep > 0) & (barsFromBullSweep <= fvgLookback))
    
    # HTF S/R Blocker - Zone 1
    # Calculate pivot highs/lows for HTF
    def rolling_pivothigh(series, length):
        result = pd.Series(np.nan, index=series.index)
        for i in range(length - 1, len(series)):
            window = series.iloc[i - length + 1:i + 1]
            if window.idxmax() == series.index[i]:
                result.iloc[i] = series.iloc[i]
        return result
    
    def rolling_pivotlow(series, length):
        result = pd.Series(np.nan, index=series.index)
        for i in range(length - 1, len(series)):
            window = series.iloc[i - length + 1:i + 1]
            if window.idxmin() == series.index[i]:
                result.iloc[i] = series.iloc[i]
        return result
    
    ph1_series = rolling_pivothigh(df['high'], htfPivotLen1)
    pl1_series = rolling_pivotlow(df['low'], htfPivotLen1)
    
    # Forward fill to get last non-na value
    lastRes1 = ph1_series.ffill()
    lastSup1 = pl1_series.ffill()
    
    nearRes1 = useHTF1 & lastRes1.notna() & (lastRes1 >= df['close']) & ((lastRes1 - df['close']) / mintick <= srBlockTicks1)
    nearSup1 = useHTF1 & lastSup1.notna() & (df['close'] >= lastSup1) & ((df['close'] - lastSup1) / mintick <= srBlockTicks1)
    
    # HTF Zone 2
    ph2_series = rolling_pivothigh(df['high'], htfPivotLen2)
    pl2_series = rolling_pivotlow(df['low'], htfPivotLen2)
    lastRes2 = ph2_series.ffill()
    lastSup2 = pl2_series.ffill()
    
    nearRes2 = useHTF2 & lastRes2.notna() & (lastRes2 >= df['close']) & ((lastRes2 - df['close']) / mintick <= srBlockTicks2)
    nearSup2 = useHTF2 & lastSup2.notna() & (df['close'] >= lastSup2) & ((df['close'] - lastSup2) / mintick <= srBlockTicks2)
    
    # Block entries near S/R
    blockedLong = nearRes1 | nearRes2
    blockedShort = nearSup1 | nearSup2
    
    # Final signals
    longSignal = finalBear_base & ~blockedLong
    shortSignal = finalBull_base & ~blockedShort
    
    # Generate entries
    entries = []
    trade_num = 1
    
    # Get numeric timestamp for output
    df_reset = df.reset_index()
    df_reset['ts_num'] = (df_reset['index'] - pd.Timestamp('1970-01-01', tz='UTC')) // pd.Timedelta('1s')
    
    for i in range(len(df_reset)):
        if longSignal.iloc[i]:
            entry_ts = int(df_reset['ts_num'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': df_reset['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df_reset['close'].iloc[i],
                'raw_price_b': df_reset['close'].iloc[i]
            })
            trade_num += 1
        
        if shortSignal.iloc[i]:
            entry_ts = int(df_reset['ts_num'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': df_reset['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df_reset['close'].iloc[i],
                'raw_price_b': df_reset['close'].iloc[i]
            })
            trade_num += 1
    
    return entries