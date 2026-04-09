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
    results = []
    trade_num = 1
    
    # Calculate ATR (Wilder RSI style)
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - df['close'].shift(1))
    tr3 = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr1 = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    atr20 = tr.ewm(alpha=1/20, min_periods=20, adjust=False).mean()
    atr2 = atr20 / 1.5
    
    # Calculate SMA for volume and close
    vol_sma9 = df['volume'].rolling(9).mean()
    loc = df['close'].rolling(54).mean()
    loc_prev = loc.shift(1)
    loc2 = loc > loc_prev
    
    # Zigzag calculation
    zigzag = []
    dir_arr = pd.Series(0.0, index=df.index)
    ph = pd.Series(np.nan, index=df.index)
    pl = pd.Series(np.nan, index=df.index)
    newbar = df['time'].diff() != 0
    
    for i in df.index:
        if newbar.iloc[i]:
            prd = 2
            bi_val = df.index[df['time'] <= df['time'].iloc[i]][-(prd):][0] if len(df.index[df['time'] <= df['time'].iloc[i]]) >= prd else df.index[0]
            bi = bi_val
            length = i - bi + 1
            if length >= 1:
                highest_idx = df['high'].iloc[max(0, i-length+1):i+1].idxmax()
                if df['high'].iloc[highest_idx] == df['high'].iloc[i]:
                    ph.iloc[i] = df['high'].iloc[i]
                lowest_idx = df['low'].iloc[max(0, i-length+1):i+1].idxmin()
                if df['low'].iloc[lowest_idx] == df['low'].iloc[i]:
                    pl.iloc[i] = df['low'].iloc[i]
        if pd.notna(ph.iloc[i]) and pd.isna(pl.iloc[i]):
            dir_arr.iloc[i] = 1
        elif pd.isna(ph.iloc[i]) and pd.notna(pl.iloc[i]):
            dir_arr.iloc[i] = -1
        else:
            dir_arr.iloc[i] = dir_arr.iloc[i-1] if i > 0 else 0
    
    # Calculate Fibonacci levels
    fib_0_arr = pd.Series(np.nan, index=df.index)
    fib_50_arr = pd.Series(np.nan, index=df.index)
    fib_1_arr = pd.Series(np.nan, index=df.index)
    
    for i in df.index:
        if len(zigzag) >= 6:
            fib_0_arr.iloc[i] = zigzag[2]
            fib_1_arr.iloc[i] = zigzag[4]
            diff = fib_1_arr.iloc[i] - fib_0_arr.iloc[i]
            fib_50_arr.iloc[i] = fib_0_arr.iloc[i] + diff * 0.5
        if pd.notna(ph.iloc[i]) or pd.notna(pl.iloc[i]):
            if dir_arr.iloc[i] != dir_arr.iloc[i-1] if i > 0 else True:
                zigzag.insert(0, df['high'].iloc[i] if dir_arr.iloc[i] == 1 else df['low'].iloc[i])
                zigzag.insert(0, i)
                if len(zigzag) > 100:
                    zigzag.pop()
                    zigzag.pop()
            else:
                if len(zigzag) > 0:
                    val = df['high'].iloc[i] if dir_arr.iloc[i] == 1 else df['low'].iloc[i]
                    if (dir_arr.iloc[i] == 1 and val > zigzag[0]) or (dir_arr.iloc[i] == -1 and val < zigzag[0]):
                        zigzag[0] = val
                        zigzag[1] = i
    
    # Filters
    volfilt = vol_sma9 * 1.5
    volfilt_check = df['volume'].shift(1) > volfilt
    atrfilt_cond1 = df['low'] - df['high'].shift(2) > atr2
    atrfilt_cond2 = df['low'].shift(2) - df['high'] > atr2
    atrfilt = atrfilt_cond1 | atrfilt_cond2
    locfiltb = loc2
    locfilts = ~loc2
    
    # FVG detection
    bull_fvg = (df['low'] > df['high'].shift(2)) & (df['close'].shift(1) > df['high'].shift(2))
    bear_fvg = (df['high'] < df['low'].shift(2)) & (df['close'].shift(1) < df['low'].shift(2))
    
    # BPR variables
    lookback_bars = 12
    bulltap = pd.Series(0, index=df.index)
    beartap = pd.Series(0, index=df.index)
    bullfvghigh = pd.Series(np.nan, index=df.index)
    bearfvglow = pd.Series(np.nan, index=df.index)
    bullstop = pd.Series(np.nan, index=df.index)
    bearstop = pd.Series(np.nan, index=df.index)
    
    # Trading window detection
    london_start1 = pd.Timestamp('07:45').time()
    london_end1 = pd.Timestamp('09:45').time()
    london_start2 = pd.Timestamp('14:45').time()
    london_end2 = pd.Timestamp('16:45').time()
    
    # Detect new day
    timestamps = pd.to_datetime(df['time'], unit='s', utc=True)
    newday = timestamps.dt.date != timestamps.dt.date.shift(1)
    newday.iloc[0] = True
    
    # Entry conditions as boolean series
    longCondition1 = bullfvghigh.notna() & (df['low'] < bullfvghigh) & (df['low'].shift(1) >= bullfvghigh)
    longCondition2 = bulltap == 1
    longCondition4 = df['close'] > fib_0_arr
    longCondition5 = fib_50_arr.notna() & fib_0_arr.notna() & fib_1_arr.notna()
    
    shortCondition1 = bearfvglow.notna() & (df['high'] > bearfvglow) & (df['high'].shift(1) <= bearfvglow)
    shortCondition2 = beartap == 1
    shortCondition4 = df['close'] < fib_1_arr
    shortCondition5 = fib_50_arr.notna() & fib_0_arr.notna() & fib_1_arr.notna()
    
    isBullishLeg = False
    isBearishLeg = False
    bullishFVG = False
    bearishFVG = False
    
    # Iterate through bars
    for i in df.index:
        ts = int(df['time'].iloc[i])
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        time_only = dt.time()
        
        in_window = (london_start1 <= time_only < london_end1) or (london_start2 <= time_only < london_end2)
        
        # BPR logic
        bull_since = 999
        bear_since = 999
        if i > 0:
            for j in range(1, min(i, 50)):
                if bull_fvg.iloc[i-j]:
                    bull_since = j
                    break
            for j in range(1, min(i, 50)):
                if bear_fvg.iloc[i-j]:
                    bear_since = j
                    break
        
        bull_cond_1 = bull_fvg.iloc[i] and bull_since <= lookback_bars
        bear_cond_1 = bear_fvg.iloc[i] and bear_since <= lookback_bars
        
        # Bullish FVG detection for entry trigger
        bull_fvg_detected = df['low'].iloc[i] > df['high'].iloc[i-2] and df['close'].iloc[i-1] > df['high'].iloc[i-2]
        bear_fvg_detected = df['high'].iloc[i] < df['low'].iloc[i-2] and df['close'].iloc[i-1] < df['low'].iloc[i-2]
        
        # FVG setup detection (sets bullfvghigh/bearfvglow)
        if bull_cond_1 and not pd.isna(fib_50_arr.iloc[i]) and not pd.isna(fib_0_arr.iloc[i]):
            if df['close'].iloc[i] < fib_50_arr.iloc[i] and df['close'].iloc[i] > fib_0_arr.iloc[i]:
                isBullishLeg = True
                isBearishLeg = False
                bullfvghigh.iloc[i] = df['low'].iloc[i]
                bullstop.iloc[i] = df['low'].iloc[i-2] if i >= 2 else np.nan
                bulltap.iloc[i] = 0
                bullishFVG = True
        
        if bear_cond_1 and not pd.isna(fib_50_arr.iloc[i]) and not pd.isna(fib_1_arr.iloc[i]):
            if df['close'].iloc[i] > fib_50_arr.iloc[i] and df['close'].iloc[i] < fib_1_arr.iloc[i]:
                isBearishLeg = True
                isBullishLeg = False
                bearfvglow.iloc[i] = df['high'].iloc[i]
                bearstop.iloc[i] = df['high'].iloc[i-2] if i >= 2 else np.nan
                beartap.iloc[i] = 0
                bearishFVG = True
        
        # Track FVG taps
        if not pd.isna(bullfvghigh.iloc[i-1]) and df['low'].iloc[i] < bullfvghigh.iloc[i-1] and df['low'].iloc[i-1] >= bullfvghigh.iloc[i-1]:
            bulltap.iloc[i] = bulltap.iloc[i-1] + 1
        else:
            bulltap.iloc[i] = bulltap.iloc[i-1] if i > 0 else 0
        
        if not pd.isna(bearfvglow.iloc[i-1]) and df['high'].iloc[i] > bearfvglow.iloc[i-1] and df['high'].iloc[i-1] <= bearfvglow.iloc[i-1]:
            beartap.iloc[i] = beartap.iloc[i-1] + 1
        else:
            beartap.iloc[i] = beartap.iloc[i-1] if i > 0 else 0
        
        # Carry forward
        if i > 0:
            if pd.isna(bullfvghigh.iloc[i]):
                bullfvghigh.iloc[i] = bullfvghigh.iloc[i-1]
            if pd.isna(bearfvglow.iloc[i]):
                bearfvglow.iloc[i] = bearfvglow.iloc[i-1]
        
        # Entry conditions check
        longCondition3 = in_window
        shortCondition3 = in_window
        
        longAllConditions = (longCondition1.iloc[i] if hasattr(longCondition1, '__getitem__') else longCondition1) and longCondition2.iloc[i] and longCondition3 and longCondition4.iloc[i] and longCondition5.iloc[i]
        shortAllConditions = (shortCondition1.iloc[i] if hasattr(shortCondition1, '__getitem__') else shortCondition1) and shortCondition2.iloc[i] and shortCondition3 and shortCondition4.iloc[i] and shortCondition5.iloc[i]
        
        if longAllConditions:
            entry_price = float(df['close'].iloc[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': dt.isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if shortAllConditions:
            entry_price = float(df['close'].iloc[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': dt.isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return results