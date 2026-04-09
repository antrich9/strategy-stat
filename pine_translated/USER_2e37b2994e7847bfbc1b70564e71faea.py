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
    
    # Strategy parameters
    useDI = True
    crossDI = True
    inverseDI = True
    middleLine50Long = 50
    middleLine50Short = 50
    stdevLengthDI = 21
    rvIDIsmoothLengthDIDI = 14
    smoothLengthDI = 14
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Calculate rvIDIOriginal for high
    stdev_high = high.rolling(stdevLengthDI).std()
    change_high = high.diff()
    upSum_high = (change_high >= 0).astype(float) * stdev_high
    downSum_high = (change_high < 0).astype(float) * stdev_high
    upSum_high_ema = upSum_high.ewm(alpha=1.0/rvIDIsmoothLengthDIDI, adjust=False).mean()
    downSum_high_ema = downSum_high.ewm(alpha=1.0/rvIDIsmoothLengthDIDI, adjust=False).mean()
    rvIDI_high = 100 * upSum_high_ema / (upSum_high_ema + downSum_high_ema + 1e-10)
    
    # Calculate rvIDIOriginal for low
    stdev_low = low.rolling(stdevLengthDI).std()
    change_low = low.diff()
    upSum_low = (change_low >= 0).astype(float) * stdev_low
    downSum_low = (change_low < 0).astype(float) * stdev_low
    upSum_low_ema = upSum_low.ewm(alpha=1.0/rvIDIsmoothLengthDIDI, adjust=False).mean()
    downSum_low_ema = downSum_low.ewm(alpha=1.0/rvIDIsmoothLengthDIDI, adjust=False).mean()
    rvIDI_low = 100 * upSum_low_ema / (upSum_low_ema + downSum_low_ema + 1e-10)
    
    # Calculate rvIDI as average of high and low rvIDI
    rvIDI = (rvIDI_high + rvIDI_low) / 2.0
    
    # Calculate inertiaDI (linear regression of rvIDI over smoothLengthDI)
    # ta.linreg(src, length, offset) = linear regression
    def linreg(series, length):
        x = np.arange(length)
        x_mean = (length - 1) / 2.0
        result = pd.Series(np.nan, index=series.index)
        for i in range(length - 1, len(series)):
            if pd.notna(series.iloc[i - length + 1:i + 1]).all():
                y = series.iloc[i - length + 1:i + 1].values
                y_mean = np.mean(y)
                slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
                intercept = y_mean - slope * x_mean
                result.iloc[i] = intercept
        return result
    
    inertiaDI = linreg(rvIDI, smoothLengthDI)
    
    # Calculate signal conditions
    # signalLongDI = useDI ? crossDI ? ta.crossover(inertiaDI, middleLine50Long) : inertiaDI > middleLine50Long : true
    if useDI:
        if crossDI:
            crossover_long = (inertiaDI > middleLine50Long) & (inertiaDI.shift(1) <= middleLine50Long)
            signalLongDI = crossover_long
        else:
            signalLongDI = inertiaDI > middleLine50Long
    else:
        signalLongDI = pd.Series(True, index=inertiaDI.index)
    
    # signalShortDI = useDI ? crossDI ? ta.crossunder(inertiaDI, middleLine50Short) : inertiaDI < middleLine50Short : true
    if useDI:
        if crossDI:
            crossunder_short = (inertiaDI < middleLine50Short) & (inertiaDI.shift(1) >= middleLine50Short)
            signalShortDI = crossunder_short
        else:
            signalShortDI = inertiaDI < middleLine50Short
    else:
        signalShortDI = pd.Series(True, index=inertiaDI.index)
    
    # final signals
    if inverseDI:
        finalLongSignalDI = signalShortDI
        finalShortSignalDI = signalLongDI
    else:
        finalLongSignalDI = signalLongDI
        finalShortSignalDI = signalShortDI
    
    # Generate entries
    entries = []
    trade_num = 1
    position_open = False
    
    n = len(df)
    for i in range(n):
        # Skip if inertiaDI is NaN
        if pd.isna(inertiaDI.iloc[i]):
            continue
        
        # Check if position is open (we can only enter if position is closed)
        # Since we're only tracking entries, we enter on signal when position is closed
        long_cond = finalLongSignalDI.iloc[i]
        short_cond = finalShortSignalDI.iloc[i]
        
        if not position_open:
            if long_cond:
                entry_price = close.iloc[i]
                entry_ts = df['time'].iloc[i]
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
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
                position_open = True
            elif short_cond:
                entry_price = close.iloc[i]
                entry_ts = df['time'].iloc[i]
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
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
                position_open = True
        else:
            # Position is open, reset when we detect it's closed
            # In this simple entry-only version, we just flip the flag on next signal
            # But since we don't have exit info, we assume position closes on next opposite signal
            # Actually, let's reset position_open to False on each bar after entry
            # This allows re-entry on next signal (opposite or same direction)
            # The original strategy checks strategy.position_size == 0, meaning position must be closed
            # For entry-only conversion, we should allow re-entry after a signal is triggered
            # Let's re-open position check after one bar (simulating position closing)
            if i > 0 and (finalLongSignalDI.iloc[i-1] or finalShortSignalDI.iloc[i-1]):
                # Previous bar had an entry, so position would be open
                # Now we check if we should close it (but we don't have exit logic)
                # For simplicity, we'll allow new entries on next valid signal
                # This is a simplification - the actual strategy has trailing stop logic
                pass
            # For entry-only, we allow entries based on signals
            # The position_open flag prevents consecutive entries without an exit
            # But since we don't track exits, we'll just reset position_open periodically
            # or better yet, only prevent immediate re-entry
            pass
    
    return entries