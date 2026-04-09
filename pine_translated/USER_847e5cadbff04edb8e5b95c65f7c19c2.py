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
    # Default parameters from Pine Script
    useMLB = True
    useMLBInverse = False
    useMLBBackground = True
    fast_lengthMLB = 14
    slow_lengthMLB = 28
    signalMLB_lengthMLB = 11
    source_typeMLB = 'ZLEMA'
    signalMLB_typeMLB = 'ZLEMA'
    use_lagMLBMLB = True
    gammaMLB = 0.02

    close = df['close'].copy()

    # ZLEMA function
    def get_zlema(src, length):
        lag = int(np.floor((length - 1) / 2))
        zlema = (src + src - src.shift(lag)).ewm(span=length, adjust=False).mean()
        return zlema

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

    # getMA function
    def get_ma(src, length, ma_type):
        if ma_type == 'SMA':
            return src.rolling(length).mean()
        elif ma_type == 'EMA':
            return src.ewm(span=length, adjust=False).mean()
        elif ma_type == 'ZLEMA':
            return get_zlema(src, length)
        return pd.Series(0.0, index=src.index)

    # Calculate MAs
    fast_ma = get_ma(close, fast_lengthMLB, source_typeMLB)
    slow_ma = get_ma(close, slow_lengthMLB, source_typeMLB)
    
    # MACD calculation
    macd = fast_ma - slow_ma
    if use_lagMLBMLB:
        macd = laguerre(gammaMLB, macd)

    # Leader calculation
    fast_leader_src = close - fast_ma
    slow_leader_src = close - slow_ma
    fast_leader = get_ma(fast_leader_src, fast_lengthMLB, source_typeMLB)
    slow_leader = get_ma(slow_leader_src, slow_lengthMLB, source_typeMLB)
    
    macd_leader = fast_ma + fast_leader - (slow_ma + slow_leader)
    if use_lagMLBMLB:
        macd_leader = laguerre(gammaMLB, macd_leader)

    # Signal line
    signal = get_ma(macd, signalMLB_lengthMLB, signalMLB_typeMLB)
    
    # Histogram
    hist = macd - signal

    # Long/Short signal conditions
    long_base = (macd.shift(1) <= macd_leader.shift(1)) & (macd > macd_leader)
    short_base = (macd.shift(1) >= macd_leader.shift(1)) & (macd < macd_leader)

    if useMLB:
        if useMLBBackground:
            signalLongMLB = (hist > 0) & long_base
            signalShortMLB = (hist < 0) & short_base
        else:
            signalLongMLB = long_base
            signalShortMLB = short_base
    else:
        signalLongMLB = pd.Series(True, index=close.index)
        signalShortMLB = pd.Series(True, index=close.index)

    # Apply inverse
    if useMLBInverse:
        finalLongSignalMLB = signalShortMLB
        finalShortSignalMLB = signalLongMLB
    else:
        finalLongSignalMLB = signalLongMLB
        finalShortSignalMLB = signalShortMLB

    # Shift signals for crossover detection (previous bar)
    finalLongSignal_prev = finalLongSignalMLB.shift(1).fillna(False).astype(bool)
    finalShortSignal_prev = finalShortSignalMLB.shift(1).fillna(False).astype(bool)

    results = []
    trade_num = 1

    # Iterate through bars starting from index 1 (need previous bar for crossover)
    for i in range(1, len(df)):
        # Check long entry condition
        if finalLongSignalMLB.iloc[i] and not finalLongSignalSignal_prev.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
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

        # Check short entry condition
        if finalShortSignalMLB.iloc[i] and not finalShortSignal_prev.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
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

    return results