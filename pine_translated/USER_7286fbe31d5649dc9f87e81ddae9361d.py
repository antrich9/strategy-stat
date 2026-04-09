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
    high = df['high']
    low = df['low']
    close = df['close']
    open_col = df['open']
    
    # Wilder RSI manual implementation (for reference)
    def wilder_rsi(src, length):
        delta = src.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    # Wilder ATR manual implementation
    def wilder_atr(high, low, close, length):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
        return atr
    
    # Vortex Indicator
    period_vi = 14
    VMP = np.abs(high - low.shift(1)).rolling(period_vi).sum()
    VMM = np.abs(low - high.shift(1)).rolling(period_vi).sum()
    STR = wilder_atr(high, low, close, period_vi).rolling(period_vi).sum()
    VIP = VMP / STR
    VIM = VMM / STR
    
    # Stiffness parameters
    maLengthStiffness = 100
    stiffLength = 60
    stiffSmooth = 3
    thresholdStiffness = 90
    
    # Stiffness calculation
    boundStiffness = close.rolling(maLengthStiffness).mean() - 0.2 * close.rolling(maLengthStiffness).std()
    sumAboveStiffness = (close > boundStiffness).rolling(stiffLength).sum()
    stiffness = (sumAboveStiffness * 100 / stiffLength).ewm(span=stiffSmooth, adjust=False).mean()
    
    signalStiffness = stiffness > thresholdStiffness
    
    # Zero Lag MACD parameters
    fast_lengthMLB = 14
    slow_lengthMLB = 28
    signalMLB_lengthMLB = 11
    gammaMLB = 0.02
    source_typeMLB = 'ZLEMA'
    signalMLB_typeMLB = 'EMA'
    use_lagMLBMLB = True
    
    # ZLEMA calculation
    def getZLEMA(src, length):
        lag = int((length - 1) / 2)
        return src.ewm(span=length, adjust=False).mean() + src.ewm(span=length, adjust=False).mean() - src.shift(lag).ewm(span=length, adjust=False).mean()
    
    # Laguerre function
    def laguerre(g, p):
        L0 = pd.Series(0.0, index=p.index)
        L1 = pd.Series(0.0, index=p.index)
        L2 = pd.Series(0.0, index=p.index)
        L3 = pd.Series(0.0, index=p.index)
        for i in range(1, len(p)):
            L0.iloc[i] = (1 - g) * p.iloc[i] + g * L0.iloc[i-1]
            L1.iloc[i] = -g * L0.iloc[i] + L0.iloc[i-1] + g * L1.iloc[i-1]
            L2.iloc[i] = -g * L1.iloc[i] + L1.iloc[i-1] + g * L2.iloc[i-1]
            L3.iloc[i] = -g * L2.iloc[i] + L2.iloc[i-1] + g * L3.iloc[i-1]
        return (L0 + 2 * L1 + 2 * L2 + L3) / 6
    
    def getMA(src, length, ma_type):
        if ma_type == 'SMA':
            return src.rolling(length).mean()
        elif ma_type == 'EMA':
            return src.ewm(span=length, adjust=False).mean()
        elif ma_type == 'ZLEMA':
            return getZLEMA(src, length)
        return src.ewm(span=length, adjust=False).mean()
    
    srcMLB = close
    fast_maMLB = getMA(srcMLB, fast_lengthMLB, source_typeMLB)
    slow_maMLB = getMA(srcMLB, slow_lengthMLB, source_typeMLB)
    macdMLB = fast_maMLB - slow_maMLB
    if use_lagMLBMLB:
        macdMLB = laguerre(gammaMLB, macdMLB)
    
    fast_leader = getMA(srcMLB - fast_maMLB, fast_lengthMLB, source_typeMLB)
    slow_leader = getMA(srcMLB - slow_maMLB, slow_lengthMLB, source_typeMLB)
    macdMLB_leader = fast_maMLB + fast_leader - (slow_maMLB + slow_leader)
    if use_lagMLBMLB:
        macdMLB_leader = laguerre(gammaMLB, macdMLB_leader)
    
    signalMLB = getMA(macdMLB, signalMLB_lengthMLB, signalMLB_typeMLB)
    histMLB = macdMLB - signalMLB
    
    # SuperTrend parameters
    Periods = 10
    src = (high + low) / 2
    Multiplier = 3.0
    
    atr_st = wilder_atr(high, low, close, Periods)
    up = src - Multiplier * atr_st
    dn = src + Multiplier * atr_st
    
    trend = pd.Series(1, index=close.index)
    for i in range(1, len(close)):
        if not np.isnan(up.iloc[i]) and not np.isnan(up.iloc[i-1]):
            up.iloc[i] = close.iloc[i-1] > up.iloc[i-1] and up.iloc[i] < up.iloc[i-1] and up.iloc[i] or max(up.iloc[i], up.iloc[i-1])
        if not np.isnan(dn.iloc[i]) and not np.isnan(dn.iloc[i-1]):
            dn.iloc[i] = close.iloc[i-1] < dn.iloc[i-1] and dn.iloc[i] > dn.iloc[i-1] and dn.iloc[i] or min(dn.iloc[i], dn.iloc[i-1])
        if trend.iloc[i-1] == -1 and close.iloc[i] > dn.iloc[i-1]:
            trend.iloc[i] = 1
        elif trend.iloc[i-1] == 1 and close.iloc[i] < up.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
    
    buySignal = (trend == 1) & (trend.shift(1) == -1)
    sellSignal = (trend == -1) & (trend.shift(1) == 1)
    
    # Second SuperTrend
    Periods1 = 10
    src1 = (high + low) / 2
    Multiplier1 = 3.0
    
    atr_st1 = wilder_atr(high, low, close, Periods1)
    up2 = src1 - Multiplier1 * atr_st1
    dn2 = src1 + Multiplier1 * atr_st1
    
    trend1 = pd.Series(1, index=close.index)
    for i in range(1, len(close)):
        if not np.isnan(up2.iloc[i]) and not np.isnan(up2.iloc[i-1]):
            up2.iloc[i] = close.iloc[i-1] > up2.iloc[i-1] and up2.iloc[i] < up2.iloc[i-1] and up2.iloc[i] or max(up2.iloc[i], up2.iloc[i-1])
        if not np.isnan(dn2.iloc[i]) and not np.isnan(dn2.iloc[i-1]):
            dn2.iloc[i] = close.iloc[i-1] < dn2.iloc[i-1] and dn2.iloc[i] > dn2.iloc[i-1] and dn2.iloc[i] or min(dn2.iloc[i], dn2.iloc[i-1])
        if trend1.iloc[i-1] == -1 and close.iloc[i] > dn2.iloc[i-1]:
            trend1.iloc[i] = 1
        elif trend1.iloc[i-1] == 1 and close.iloc[i] < up2.iloc[i-1]:
            trend1.iloc[i] = -1
        else:
            trend1.iloc[i] = trend1.iloc[i-1]
    
    buySignal1 = (trend1 == 1) & (trend1.shift(1) == -1)
    sellSignal1 = (trend1 == -1) & (trend1.shift(1) == 1)
    
    # Zero Lag MACD (end of script)
    source = close
    fastLength = 12
    slowLength = 26
    signalLength = 9
    
    ema1 = source.ewm(span=fastLength, adjust=False).mean()
    ema2 = ema1.ewm(span=fastLength, adjust=False).mean()
    differenceFast = ema1 - ema2
    zerolagEMA = ema1 + differenceFast
    demaFast = (2 * ema1) - ema2
    
    emas1 = source.ewm(span=slowLength, adjust=False).mean()
    emas2 = emas1.ewm(span=slowLength, adjust=False).mean()
    differenceSlow = emas1 - emas2
    zerolagslowMA = emas1 + differenceSlow
    demaSlow = (2 * emas1) - emas2
    
    ZeroLagMACD = demaFast - demaSlow
    
    emasig1 = ZeroLagMACD.ewm(span=signalLength, adjust=False).mean()
    emasig2 = emasig1.ewm(span=signalLength, adjust=False).mean()
    signal = (2 * emasig1) - emasig2
    
    hist = ZeroLagMACD - signal
    
    # Build conditions
    cond_long = signalStiffness & buySignal
    cond_short = signalStiffness & sellSignal
    
    # MLB long condition: histMLB > 0 and macdMLB crosses below macdMLB_leader
    macd_crossunder_mlb = (macdMLB < macdMLB_leader) & (macdMLB.shift(1) >= macdMLB_leader.shift(1))
    cond_long_mlb = histMLB > 0 & macd_crossunder_mlb
    
    # MLB short condition: histMLB < 0 and macdMLB crosses above macdMLB_leader
    macd_crossover_mlb = (macdMLB > macdMLB_leader) & (macdMLB.shift(1) <= macdMLB_leader.shift(1))
    cond_short_mlb = histMLB < 0 & macd_crossover_mlb
    
    # ZLMACD long condition: hist > 0 and hist crosses below signal
    zlmacd_crossunder = (hist < signal) & (hist.shift(1) >= signal.shift(1))
    cond_long_zlmacd = hist > 0 & zlmacd_crossunder
    
    # ZLMACD short condition: hist < 0 and hist crosses above signal
    zlmacd_crossover = (hist > signal) & (hist.shift(1) <= signal.shift(1))
    cond_short_zlmacd = hist < 0 & zlmacd_crossover
    
    # Skip bars with NaN
    valid_mask = (
        (~stiffness.isna()) &
        (~buySignal.isna()) &
        (~sellSignal.isna()) &
        (~macdMLB.isna()) &
        (~macdMLB_leader.isna()) &
        (~histMLB.isna()) &
        (~trend.isna()) &
        (~ZeroLagMACD.isna()) &
        (~signal.isna()) &
        (~hist.isna())
    )
    
    cond_long = cond_long & valid_mask
    cond_short = cond_short & valid_mask
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        direction = None
        if cond_long.iloc[i]:
            direction = 'long'
        elif cond_short.iloc[i]:
            direction = 'short'
        
        if direction:
            entry_price = close.iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries