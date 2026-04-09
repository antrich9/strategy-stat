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
    atrLength = 14
    atrMultiplier = 2.0
    tradeDirection = "Both"
    dpoLength = 20
    adxThreshold = 25.0
    emaLength = 200
    adxLength = 14
    input_retSince = 2
    input_retValid = 2
    bb = 20
    
    # Calculate EMA
    ema200 = df['close'].ewm(span=emaLength, adjust=False).mean()
    
    # Calculate ATR (Wilder's method)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = tr.ewm(alpha=1/atrLength, adjust=False).mean()
    
    # Calculate DPO
    dpoDisplace = int(np.ceil(dpoLength / 2)) + 1
    smaForDPO = df['close'].rolling(dpoLength).mean()
    dpo = df['close'].shift(dpoDisplace) - smaForDPO
    
    # Get previous day high and low (assuming daily data)
    prevDayHigh = df['high'].shift(1)
    prevDayLow = df['low'].shift(1)
    
    # Detect if price sweeps previous day's high/low
    pdhSwept = df['high'] > prevDayHigh
    pdlSwept = df['low'] < prevDayLow
    
    # Define trend direction
    bullishTrend = df['close'] > ema200
    bearishTrend = df['close'] < ema200
    
    # Detect Breakouts
    breakoutLong = (df['high'] > prevDayHigh) & (df['close'] > prevDayHigh)
    breakoutShort = (df['low'] < prevDayLow) & (df['close'] < prevDayLow)
    
    # Calculate ADX (Wilder's method)
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0).where(plus_dm > 0, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0).where(minus_dm > 0, 0)
    
    tr_smooth = tr.ewm(alpha=1/adxLength, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1/adxLength, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/adxLength, adjust=False).mean()
    
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/adxLength, adjust=False).mean()
    
    # Calculate pivot points for retest detection
    pl = df['low'].rolling(window=bb+1).min().shift(-bb)
    ph = df['high'].rolling(window=bb+1).max().shift(-bb)
    pl = pl.where(pl.notna(), df['low'].rolling(window=bb+1).min().shift(-1))
    ph = ph.where(ph.notna(), df['high'].rolling(window=bb+1).max().shift(-1))
    
    # Box boundaries (simplified)
    sBot = df['low'].shift(1).rolling(bb).min()
    rTop = df['high'].shift(1).rolling(bb).max()
    
    # Pivot change detection
    pl_change = pl.diff() != 0
    ph_change = ph.diff() != 0
    
    # Retest conditions (simplified)
    sBreak = breakoutLong
    rBreak = breakoutShort
    
    s1 = (sBreak.shift(1)) & ((df['high'].shift(1) >= sBot) & (df['close'] <= sBot))
    s2 = (sBreak.shift(1)) & ((df['high'].shift(1) >= sBot) & (df['close'] >= sBot) & (df['close'] <= sBot))
    s3 = (sBreak.shift(1)) & ((df['high'].shift(1) >= sBot) & (df['high'].shift(1) <= sBot))
    s4 = (sBreak.shift(1)) & ((df['high'].shift(1) >= sBot) & (df['high'].shift(1) <= sBot) & (df['close'] < sBot))
    
    r1 = (rBreak.shift(1)) & ((df['low'].shift(1) <= rTop) & (df['close'] >= rTop))
    r2 = (rBreak.shift(1)) & ((df['low'].shift(1) <= rTop) & (df['close'] <= rTop) & (df['close'] >= rTop))
    r3 = (rBreak.shift(1)) & ((df['low'].shift(1) <= rTop) & (df['low'].shift(1) >= rTop))
    r4 = (rBreak.shift(1)) & ((df['low'].shift(1) <= rTop) & (df['low'].shift(1) >= rTop) & (df['close'] > rTop))
    
    long_retest = s1 | s2 | s3 | s4
    short_retest = r1 | r2 | r3 | r4
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(adx.iloc[i]) or adx.iloc[i] < adxThreshold:
            continue
        
        long_signal = (breakoutLong.iloc[i] and bullishTrend.iloc[i]) or (long_retest.iloc[i] and bullishTrend.iloc[i])
        short_signal = (breakoutShort.iloc[i] and bearishTrend.iloc[i]) or (short_retest.iloc[i] and bearishTrend.iloc[i])
        
        direction = None
        if tradeDirection == "Long" and long_signal:
            direction = "long"
        elif tradeDirection == "Short" and short_signal:
            direction = "short"
        elif tradeDirection == "Both":
            if long_signal:
                direction = "long"
            elif short_signal:
                direction = "short"
        
        if direction:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
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