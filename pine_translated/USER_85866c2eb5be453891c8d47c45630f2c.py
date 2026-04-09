import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # ADX Parameters
    adxLength = 14
    adxThreshold = 25
    
    # Calculate +DI and -DI
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    atr_len = 14
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder smoothed ATR
    atr = tr.ewm(alpha=1/atr_len, adjust=False).mean()
    
    # Wilder smoothed +DM and -DM
    plus_dm_smooth = plus_dm.ewm(alpha=1/adxLength, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/adxLength, adjust=False).mean()
    
    # +DI and -DI
    di_plus = 100 * plus_dm_smooth / atr
    di_minus = 100 * minus_dm_smooth / atr
    
    # DX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    
    # ADX
    adx = dx.ewm(alpha=1/adxLength, adjust=False).mean()
    
    # Hull MA parameters
    useHullMA = True
    usecolorHullMA = True
    lengthHullMA = 9
    srcHullMA = close
    
    # Hull MA calculation
    half_length = int(lengthHullMA / 2)
    sqrt_length = int(np.floor(np.sqrt(lengthHullMA)))
    
    wma_half = (2 * srcHullMA.rolling(half_length).mean()) - srcHullMA.rolling(lengthHullMA).mean()
    hullmaHullMA = (2 * wma_half).rolling(sqrt_length).mean()
    
    # sigHullMA
    sigHullMA = (hullmaHullMA > hullmaHullMA.shift(1)).astype(int) - (hullmaHullMA <= hullmaHullMA.shift(1)).astype(int)
    
    # T3 parameters
    useT3 = True
    crossT3 = True
    inverseT3 = False
    lengthT3 = 5
    factorT3 = 0.7
    highlightMovementsT3 = True
    srcT3 = close
    
    # T3 calculation
    ema1 = srcT3.ewm(span=lengthT3, adjust=False).mean()
    ema2 = ema1.ewm(span=lengthT3, adjust=False).mean()
    gd = ema1 * (1 + factorT3) - ema2 * factorT3
    
    ema1_2 = gd.ewm(span=lengthT3, adjust=False).mean()
    ema2_2 = ema1_2.ewm(span=lengthT3, adjust=False).mean()
    gd2 = ema1_2 * (1 + factorT3) - ema2_2 * factorT3
    
    ema1_3 = gd2.ewm(span=lengthT3, adjust=False).mean()
    ema2_3 = ema1_3.ewm(span=lengthT3, adjust=False).mean()
    t3 = ema1_3 * (1 + factorT3) - ema2_3 * factorT3
    
    # t3Signals
    t3Signals = (t3 > t3.shift(1)).astype(int) - (t3 <= t3.shift(1)).astype(int)
    
    # Kalman Filter
    useKalmanFilter = True
    q = 0.001
    r = 0.001
    
    kalmanPrice = pd.Series(index=df.index, dtype=float)
    x = None
    p = 1.0
    
    for i in range(len(df)):
        if pd.isna(kalmanPrice.iloc[i-1]) if i > 0 else True:
            x = close.iloc[i]
        else:
            x = kalmanPrice.iloc[i-1]
        
        xPredicted = x
        pPredicted = p + q
        k = pPredicted / (pPredicted + r)
        x = xPredicted + k * (close.iloc[i] - xPredicted)
        p = (1 - k) * pPredicted
        kalmanPrice.iloc[i] = x
    
    # Entry conditions
    signalHullMALong = (sigHullMA > 0) & (close > hullmaHullMA)
    basicLongCondition = (t3Signals > 0) & (close > t3)
    t3SignalsLong = basicLongCondition if highlightMovementsT3 else (close > t3)
    t3SignalsLongCross = (~t3SignalsLong.shift(1).fillna(False)) & t3SignalsLong if crossT3 else t3SignalsLong
    t3SignalsLongFinal = (~t3SignalsLongCross) if inverseT3 else t3SignalsLongCross
    kalmanLongCondition = (close > kalmanPrice) if useKalmanFilter else pd.Series(True, index=df.index)
    
    entryCondition = signalHullMALong & t3SignalsLongFinal & kalmanLongCondition & (adx > adxThreshold)
    
    signalHullMAShort = (sigHullMA < 0) & (close < hullmaHullMA)
    basicShortCondition = (t3Signals < 0) & (close < t3)
    t3SignalsShort = basicShortCondition if highlightMovementsT3 else (close < t3)
    t3SignalsShortCross = (~t3SignalsShort.shift(1).fillna(False)) & t3SignalsShort if crossT3 else t3SignalsShort
    t3SignalsShortFinal = (~t3SignalsShortCross) if inverseT3 else t3SignalsShortCross
    kalmanShortCondition = (close < kalmanPrice) if useKalmanFilter else pd.Series(True, index=df.index)
    
    shortEntryCondition = signalHullMAShort & t3SignalsShortFinal & kalmanShortCondition & (adx > adxThreshold)
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i == 0:
            continue
        
        if pd.isna(hullmaHullMA.iloc[i]) or pd.isna(t3.iloc[i]) or pd.isna(kalmanPrice.iloc[i]) or pd.isna(adx.iloc[i]):
            continue
        
        if shortEntryCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if entryCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries