import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    entries = []
    trade_num = 0
    
    # Parameters
    adxLength = 14
    adxThreshold = 20
    useHullMA = True
    usecolorHullMA = True
    lengthHullMA = 9
    useT3 = True
    crossT3 = True
    inverseT3 = False
    highlightMovementsT3 = True
    lengthT3 = 5
    factorT3 = 0.7
    useKalmanFilter = True
    q = 0.001
    r = 0.001
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/adxLength, adjust=False).mean()
    
    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    atr_smooth = atr.copy()
    di_plus = (plus_dm.ewm(alpha=1.0/adxLength, adjust=False).mean() / atr_smooth) * 100
    di_minus = (minus_dm.ewm(alpha=1.0/adxLength, adjust=False).mean() / atr_smooth) * 100
    
    di_sum = di_plus + di_minus
    di_sum = di_sum.replace(0, np.nan)
    di_plus_pct = (di_plus / di_sum) * 100
    di_minus_pct = (di_minus / di_sum) * 100
    
    dx = (abs(di_plus_pct - di_minus_pct) / (di_plus_pct + di_minus_pct)) * 100
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx_series = dx.ewm(alpha=1.0/adxLength, adjust=False).mean()
    
    # Hull MA
    half_length = int(lengthHullMA / 2)
    sqrt_length = int(np.sqrt(lengthHullMA))
    
    wma_half = close.rolling(half_length).mean()
    wma_full = close.rolling(lengthHullMA).mean()
    hullma_raw = 2 * wma_half - wma_full
    hullma = hullma_raw.rolling(sqrt_length).mean()
    
    sigHullMA = (hullma > hullma.shift(1)).astype(int) - (hullma < hullma.shift(1)).astype(int)
    
    # T3
    ema1 = close.ewm(span=lengthT3, adjust=False).mean()
    ema2 = ema1.ewm(span=lengthT3, adjust=False).mean()
    t3 = ema1 * (1 + factorT3) - ema2 * factorT3
    t3 = t3 * (1 + factorT3) - t3.ewm(span=lengthT3, adjust=False).mean() * factorT3
    t3 = t3 * (1 + factorT3) - t3.ewm(span=lengthT3, adjust=False).mean() * factorT3
    
    t3Signals = (t3 > t3.shift(1)).astype(int) - (t3 < t3.shift(1)).astype(int)
    
    # Kalman Filter
    kalman_state = pd.Series(np.nan, index=df.index)
    kalman_p = pd.Series(1.0, index=df.index)
    
    for i in range(len(df)):
        source = close.iloc[i]
        if i == 0:
            x = source
            p = 1.0
        else:
            x = kalman_state.iloc[i-1] if pd.notna(kalman_state.iloc[i-1]) else source
            p = kalman_p.iloc[i-1] if pd.notna(kalman_p.iloc[i-1]) else 1.0
        
        x_pred = x
        p_pred = p + q
        k = p_pred / (p_pred + r)
        x_new = x_pred + k * (source - x_pred)
        p_new = (1 - k) * p_pred
        
        kalman_state.iloc[i] = x_new
        kalman_p.iloc[i] = p_new
    
    # Entry conditions
    signalHullMALong = (sigHullMA > 0) & (close > hullma)
    
    basicLongCondition = (t3Signals > 0) & (close > t3)
    t3SignalsLong = basicLongCondition if highlightMovementsT3 else (close > t3)
    
    t3SignalsLong_shifted = t3SignalsLong.shift(1).fillna(False).astype(bool)
    t3SignalsLongCross = (~t3SignalsLong_shifted) & (t3SignalsLong.astype(bool)) if crossT3 else t3SignalsLong.astype(bool)
    
    t3SignalsLongFinal = ~t3SignalsLongCross if inverseT3 else t3SignalsLongCross
    
    kalmanLongCondition = close > kalman_state
    
    entryCondition = t3SignalsLongFinal & kalmanLongCondition & (adx_series > adxThreshold)
    
    for i in range(1, len(df)):
        if pd.isna(hullma.iloc[i]) or pd.isna(t3.iloc[i]) or pd.isna(kalman_state.iloc[i]) or pd.isna(adx_series.iloc[i]):
            continue
        if entryCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            trade_num += 1
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
    
    return entries