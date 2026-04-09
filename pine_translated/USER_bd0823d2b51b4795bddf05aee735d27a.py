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
    
    # Input parameters
    inp1 = False  # Volume Filter
    inp2 = False  # ATR Filter
    inp3 = False  # Trend Filter
    
    # Time window calculations
    timestamps = pd.to_datetime(df['time'], unit='s', utc=True)
    london_times = timestamps.dt.tz_convert('Europe/London')
    hour = london_times.dt.hour
    minute = london_times.dt.minute
    
    # Morning window: 7:45 to 9:45 London time
    isWithinMorningWindow = (hour * 60 + minute >= 7 * 60 + 45) & (hour * 60 + minute < 9 * 60 + 45)
    
    # Afternoon window: 14:45 to 16:45 London time
    isWithinAfternoonWindow = (hour * 60 + minute >= 14 * 60 + 45) & (hour * 60 + minute < 16 * 60 + 45)
    
    # Combined trading window
    in_trading_window = isWithinMorningWindow | isWithinAfternoonWindow
    
    # Volume filter
    volfilt = volume.rolling(9).mean() * 1.5 if inp1 else pd.Series(True, index=df.index)
    volfilt = df['volume'] > volfilt if inp1 else pd.Series(True, index=df.index)
    
    # ATR filter
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean() / 1.5
    atrfilt = ((df['low'] - df['high'].shift(2)) > atr) | ((df['low'].shift(2) - df['high']) > atr) if inp2 else pd.Series(True, index=df.index)
    
    # Trend filter
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2 if inp3 else pd.Series(True, index=df.index)
    locfilts = ~loc2 if inp3 else pd.Series(True, index=df.index)
    
    # FVG conditions
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    # OB conditions (using shift for Pine Script index references)
    obUp = (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open']) & (df['close'] > df['high'].shift(1))
    obDown = (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open']) & (df['close'] < df['low'].shift(1))
    
    # FVG stack conditions
    fvgUp = df['low'] > df['high'].shift(2)
    fvgDown = df['high'] < df['low'].shift(2)
    
    # Combined conditions (Stacked OB + FVG)
    long_condition = obUp & fvgUp & in_trading_window
    short_condition = obDown & fvgDown & in_trading_window
    
    # Consecutive counters
    consecutiveBfvg = 0
    consecutiveSfvg = 0
    flagBfvg = False
    
    entries = []
    trade_num = 0
    
    valid_indices = df.index[(df['high'].notna()) & (df['low'].notna()) & (df['close'].notna())]
    
    for i in valid_indices:
        bfvg_val = bfvg.iloc[i] if i in bfvg.index else False
        sfvg_val = sfvg.iloc[i] if i in sfvg.index else False
        
        if bfvg_val:
            consecutiveBfvg += 1
            flagBfvg = True
            consecutiveSfvg = 0
        elif sfvg_val:
            consecutiveSfvg += 1
            if flagBfvg and consecutiveSfvg == 1:
                pass
            flagBfvg = False
        else:
            consecutiveBfvg = 0
            consecutiveSfvg = 0
            flagBfvg = False
        
        long_signal = long_condition.iloc[i] if i in long_condition.index else False
        short_signal = short_condition.iloc[i] if i in short_condition.index else False
        
        if long_signal:
            trade_num += 1
            ts = int(df['time'].iloc[i])
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
        
        if short_signal:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
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