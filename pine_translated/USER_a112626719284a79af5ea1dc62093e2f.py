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
    # Input parameters (matching default values from Pine Script)
    useTDFI = True
    crossTDFI = True
    inverseTDFI = True
    
    lookbackTDFI = 13
    mmaLengthTDFI = 13
    smmaLengthTDFI = 13
    nLengthTDFI = 3
    filterHighTDFI = 0.05
    filterLowTDFI = -0.05
    
    # TEMA calculation
    def tema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3
    
    # MA function for TDFI (supports EMA, WMA, SWMA, VWMA, HULL, TEMA)
    def ma_tdfi(mode, src, length):
        if mode == 'ema':
            return src.ewm(span=length, adjust=False).mean()
        elif mode == 'wma':
            weights = np.arange(1, length + 1)
            return src.rolling(length).apply(lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(), raw=True)
        elif mode == 'swma':
            return src.rolling(8).mean()  # Simple SWMA approximation
        elif mode == 'vwma':
            # VWMA requires volume, simplified here as EMA
            return src.ewm(span=length, adjust=False).mean()
        elif mode == 'hull':
            half_len = int(length / 2)
            if half_len < 1:
                half_len = 1
            wma1 = src.rolling(half_len).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)).sum() / (len(x)*(len(x)+1)/2), raw=True)
            wma2 = src.rolling(length).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)).sum() / (len(x)*(len(x)+1)/2), raw=True)
            hull = 2 * wma1 - wma2
            sqrt_len = int(np.sqrt(length) + 0.5)
            if sqrt_len < 1:
                sqrt_len = 1
            return hull.rolling(sqrt_len).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)).sum() / (len(x)*(len(x)+1)/2), raw=True)
        elif mode == 'tema':
            return tema(src, length)
        else:  # sma
            return src.rolling(length).mean()
    
    # TDFI calculation
    price_tdfi = df['close']
    mma_tdfi = ma_tdfi('ema', price_tdfi * 1000, mmaLengthTDFI)
    smma_tdfi = ma_tdfi('ema', mma_tdfi, smmaLengthTDFI)
    
    impetmma_tdfi = mma_tdfi - mma_tdfi.shift(1)
    impetsmma_tdfi = smma_tdfi - smma_tdfi.shift(1)
    divma_tdfi = (mma_tdfi - smma_tdfi).abs()
    averimpet_tdfi = (impetmma_tdfi + impetsmma_tdfi) / 2
    tdf_tdfi = (divma_tdfi ** 1) * (averimpet_tdfi ** nLengthTDFI)
    
    # Rolling max with lookback
    roll_max = tdf_tdfi.abs().rolling(window=lookbackTDFI * nLengthTDFI).max()
    signal_tdfi = tdf_tdfi / roll_max
    
    # Signal conditions
    signal_long_tdfi = signal_tdfi > filterHighTDFI
    signal_short_tdfi = signal_tdfi < filterLowTDFI
    
    # Cross conditions
    if crossTDFI:
        crossover_cond = (signal_tdfi > filterHighTDFI) & (signal_tdfi.shift(1) <= filterHighTDFI)
        crossunder_cond = (signal_tdfi < filterLowTDFI) & (signal_tdfi.shift(1) >= filterLowTDFI)
    else:
        crossover_cond = signal_long_tdfi
        crossunder_cond = signal_short_tdfi
    
    # Final signals with inverse
    if inverseTDFI:
        final_long_signal = crossunder_cond
        final_short_signal = crossover_cond
    else:
        final_long_signal = crossover_cond if useTDFI else pd.Series(True, index=signal_tdfi.index)
        final_short_signal = crossunder_cond if useTDFI else pd.Series(True, index=signal_tdfi.index)
    
    # Build conditions
    long_condition = final_long_signal
    short_condition = final_short_signal
    
    # Entry logic (strategy.position_size == 0)
    in_position = False
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if in_position:
            continue
        
        # Skip if TDFI is NaN
        if pd.isna(signal_tdfi.iloc[i]):
            continue
        
        # Long entry
        if long_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_price_guess = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1
            in_position = True
        # Short entry
        elif short_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_price_guess = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1
            in_position = True
    
    return entries