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
    
    # T3 Inputs
    lengthT3 = 5
    factor = 0.7
    srcT3 = df['close']
    
    # T3 Calculation (Tillson T3)
    ema1 = srcT3.ewm(span=lengthT3, adjust=False).mean()
    ema2 = ema1.ewm(span=lengthT3, adjust=False).mean()
    ema3 = ema2.ewm(span=lengthT3, adjust=False).mean()
    t3 = ema1 * (1 + factor) - ema2 * factor
    t3 = t3.ewm(span=lengthT3, adjust=False).mean()  # Third application
    ema4 = t3.ewm(span=lengthT3, adjust=False).mean()
    t3 = t3 * (1 + factor) - ema4 * factor
    
    # Trendilo Inputs
    srcTrendilo = df['close']
    smooth = 1
    lengthTrendilo = 50
    offset = 0.85
    sigma = 6
    bmult = 1.0
    
    # Trendilo Calculation
    pch = df['close'].diff(smooth) / df['close'] * 100
    
    # ALMA implementation
    def alma(series, length, offset, sigma):
        window = length
        m = (offset * (window - 1))
        s = sigma * window / 6
        k = np.exp(-(s * s) * 2)
        w = np.exp(-((np.arange(window) - m) ** 2) / (2 * k * k))
        w = w / w.sum()
        return pd.Series([np.nan] * (length - 1) + [np.convolve(series.values, w, mode='valid')[0]], index=series.index)
    
    avpch = pch.rolling(1).apply(lambda x: alma(pd.Series(x), lengthTrendilo, offset, sigma).iloc[0] if len(x) >= 1 else np.nan, raw=False)
    # Simplified ALMA using rolling with custom weights
    def calc_alma(series, length, offset, sigma):
        result = pd.Series(np.nan, index=series.index)
        window = length
        m = offset * (window - 1)
        s = sigma * window / 6
        k = np.exp(-(s * s) * 2)
        w = np.exp(-((np.arange(window) - m) ** 2) / (2 * k * k))
        w = w / w.sum()
        for i in range(window - 1, len(series)):
            result.iloc[i] = np.sum(series.iloc[i - window + 1:i + 1].values * w)
        return result
    
    avpch = calc_alma(pch, lengthTrendilo, offset, sigma)
    
    blength = lengthTrendilo
    rms = bmult * np.sqrt(avpch.rolling(blength).apply(lambda x: (x * x).sum() / blength, raw=True))
    cdir = pd.Series(np.where(avpch > rms, 1, np.where(avpch < -rms, -1, 0)), index=df.index)
    
    # WAE Inputs
    sensitivity = 150
    fastLength = 20
    slowLength = 40
    channelLength = 20
    mult = 2.0
    
    # WAE Calculation
    ema_fast = df['close'].ewm(span=fastLength, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slowLength, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_prev = macd.shift(1)
    t1 = (macd - macd_prev) * sensitivity
    
    sma_bb = df['close'].rolling(channelLength).mean()
    std_bb = df['close'].rolling(channelLength).std()
    bb_upper = sma_bb + mult * std_bb
    bb_lower = sma_bb - mult * std_bb
    e1 = bb_upper - bb_lower
    
    waeConditionLong = (t1 > e1) & (t1 > 0)
    waeConditionShort = (-t1 > e1) & (t1 < 0)
    
    firstWaeCandleLong = waeConditionLong & (~waeConditionLong.shift(1).fillna(False))
    firstWaeCandleShort = waeConditionShort & (~waeConditionShort.shift(1).fillna(False))
    
    # Stiffness Inputs
    useStiffness = False
    maLengthStiffness = 100
    stiffLength = 60
    stiffSmooth = 3
    thresholdStiffness = 90
    
    # Stiffness Calculation
    boundStiffness = df['close'].rolling(maLengthStiffness).mean() - 0.2 * df['close'].rolling(maLengthStiffness).std()
    sumAboveStiffness = (df['close'] > boundStiffness).rolling(stiffLength).sum()
    stiffness = (sumAboveStiffness * 100 / stiffLength).ewm(span=stiffSmooth, adjust=False).mean()
    
    signalStiffness = True  # useStiffness is False in inputs
    
    # Combined Entry Conditions
    long_condition = (df['close'] > t3) & (cdir == 1) & firstWaeCandleLong & signalStiffness
    short_condition = (df['close'] < t3) & (cdir == -1) & firstWaeCandleShort & signalStiffness
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        # Skip if required indicators are NaN
        if (pd.isna(t3.iloc[i]) or pd.isna(cdir.iloc[i]) or 
            pd.isna(stiffness.iloc[i]) or pd.isna(boundStiffness.iloc[i])):
            continue
        
        if long_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price_guess = float(df['close'].iloc[i])
            
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
            entry_price_guess = float(df['close'].iloc[i])
            
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