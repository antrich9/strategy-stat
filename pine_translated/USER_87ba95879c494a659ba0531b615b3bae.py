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
    
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    time = df['time']
    
    # Calculate indicators
    
    # Wilder ATR (using 144 for main FVG ATR)
    def wilder_atr(period):
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    # ATR for FVG detection (144 period as per script)
    atr_fvg = wilder_atr(144)
    
    # ATR for filter (20 period)
    atr_filter = wilder_atr(20) / 1.5
    
    # Volume SMA (9 period)
    vol_sma = volume.rolling(9).mean()
    
    # Volume filter: volume[1] > sma(volume, 9) * 1.5
    vol_filt = volume.shift(1) > vol_sma * 1.5
    
    # Close SMA (54 period) for trend
    close_sma = close.rolling(54).mean()
    loc2 = close_sma > close_sma.shift(1)
    
    # Bullish and Bearish Gaps
    bullG = low > high.shift(1)
    bearG = high < low.shift(1)
    
    # FVG width filter (fvgTH = 0.5)
    fvgTH = 0.5
    fvg_atr = atr_fvg * fvgTH
    
    # Main FVG conditions from script:
    # bull = (b.l - b.h[2]) > atr and b.l > b.h[2] and b.c[1] > b.h[2] and not (bullG or bullG[1])
    # bear = (b.l[2] - b.h) > atr and b.h < b.l[2] and b.c[1] < b.l[2] and not (bearG or bearG[1])
    
    bull = (low - high.shift(2)) > fvg_atr
    bull &= low > high.shift(2)
    bull &= close.shift(1) > high.shift(2)
    bull &= ~(bullG | bullG.shift(1))
    
    bear = (low.shift(2) - high) > fvg_atr
    bear &= high < low.shift(2)
    bear &= close.shift(1) < low.shift(2)
    bear &= ~(bearG | bearG.shift(1))
    
    # Additional FVG conditions (bfvg/sfvg) for consecutive tracking
    bfvg_cond = low > high.shift(2)
    bfvg_cond &= vol_filt
    bfvg_cond &= ((low - high.shift(2)) > atr_filter) | ((low.shift(2) - high) > atr_filter)
    bfvg_cond &= loc2
    
    sfvg_cond = high < low.shift(2)
    sfvg_cond &= vol_filt
    sfvg_cond &= ((low - high.shift(2)) > atr_filter) | ((low.shift(2) - high) > atr_filter)
    sfvg_cond &= ~loc2
    
    # Track consecutive FVGs
    consecutive_bfvg = pd.Series(0, index=df.index)
    consecutive_sfvg = pd.Series(0, index=df.index)
    
    for i in range(2, len(df)):
        if bfvg_cond.iloc[i-1]:
            if bfvg_cond.iloc[i-2]:
                consecutive_bfvg.iloc[i] = consecutive_bfvg.iloc[i-1] + 1
            else:
                consecutive_bfvg.iloc[i] = 1
            consecutive_sfvg.iloc[i] = 0
        elif sfvg_cond.iloc[i-1]:
            if sfvg_cond.iloc[i-2]:
                consecutive_sfvg.iloc[i] = consecutive_sfvg.iloc[i-1] + 1
            else:
                consecutive_sfvg.iloc[i] = 1
            consecutive_bfvg.iloc[i] = 0
        else:
            consecutive_bfvg.iloc[i] = 0
            consecutive_sfvg.iloc[i] = 0
    
    # Entry conditions: use bull for long, bear for short
    long_entry = bull
    short_entry = bear
    
    # Iterate and generate entries
    for i in range(len(df)):
        if i < 2:
            continue
        
        if long_entry.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = int(time.iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if short_entry.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = int(time.iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return results