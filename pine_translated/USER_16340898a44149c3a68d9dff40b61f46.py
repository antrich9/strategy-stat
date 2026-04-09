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
    
    df = df.copy()
    df['ts'] = df['time']
    df.set_index('time', inplace=True)
    
    # Create 4H timeframe data using resampling
    ohlc_4h = df[['open', 'high', 'low', 'close', 'volume']].resample('240min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    high_4h = ohlc_4h['high']
    low_4h = ohlc_4h['low']
    close_4h = ohlc_4h['close']
    volume_4h = ohlc_4h['volume']
    
    # Detect new 4H bar
    is_new_4h = pd.Series(False, index=ohlc_4h.index)
    is_new_4h.iloc[1:] = True
    
    # Volume Filter (4H data)
    volfilt_4h = volume_4h.shift(1) > volume_4h.rolling(9).mean() * 1.5
    
    # ATR Filter (4H data) - Wilder ATR
    tr_4h = pd.concat([high_4h - low_4h, (high_4h - close_4h.shift(1)).abs(), (low_4h - close_4h.shift(1)).abs()], axis=1).max(axis=1)
    atr_4h_raw = tr_4h.ewm(alpha=1/20, adjust=False).mean()
    atr_4h = atr_4h_raw / 1.5
    
    atrfilt = ((low_4h - high_4h.shift(2) > atr_4h) | (low_4h.shift(2) - high_4h > atr_4h))
    
    # Trend Filter (4H data)
    loc1 = close_4h.rolling(54).mean()
    loc2 = loc1 > loc1.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # Identify Bullish and Bearish FVGs (4H data)
    bfvg1 = (low_4h > high_4h.shift(2)) & volfilt_4h & atrfilt & locfiltb
    sfvg1 = (high_4h < low_4h.shift(2)) & volfilt_4h & atrfilt & locfilts
    
    # Chart timeframe FVG conditions
    volfilt = df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5
    
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
    atr2 = tr.ewm(alpha=1/20, adjust=False).mean() / 1.5
    atrfilt_chart = ((df['low'] - df['high'].shift(2) > atr2) | (df['low'].shift(2) - df['high'] > atr2))
    
    loc_chart = df['close'].rolling(54).mean()
    loc2_chart = loc_chart > loc_chart.shift(1)
    locfiltb_chart = loc2_chart
    locfilts_chart = ~loc2_chart
    
    bfvg_chart = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt_chart & locfiltb_chart
    sfvg_chart = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt_chart & locfilts_chart
    
    # Map 4H conditions back to chart timeframe
    # Create a mapping from chart time to 4H time
    chart_times = df.index
    four_h_times = ohlc_4h.index
    
    # Find corresponding 4H bar for each chart bar
    def get_4h_idx(chart_ts):
        idx = four_h_times.get_indexer([chart_ts], method='ffill')
        if idx[0] >= 0:
            return four_h_times[idx[0]]
        return None
    
    # Create aligned 4H series on chart timeframe index
    bfvg1_aligned = pd.Series(False, index=chart_times)
    sfvg1_aligned = pd.Series(False, index=chart_times)
    is_new_4h_aligned = pd.Series(False, index=chart_times)
    last_4h_idx = None
    
    for i, ts in enumerate(chart_times):
        four_h_ts = get_4h_idx(ts)
        if four_h_ts is not None:
            if last_4h_idx is not None and four_h_ts != last_4h_idx:
                is_new_4h_aligned.iloc[i] = True
            last_4h_idx = four_h_ts
            bfvg1_aligned.iloc[i] = bfvg1.get(four_h_ts, False)
            sfvg1_aligned.iloc[i] = sfvg1.get(four_h_ts, False)
    
    # Simulate lastFVG state machine
    entries = []
    trade_num = 1
    
    lastFVG = 0
    prev_confirmed = False
    is_first = True
    new_day = False
    
    for i in range(len(df)):
        if i == 0:
            is_first = True
        elif prev_confirmed:
            is_first = False
        else:
            is_first = (i > 0) and (df.index[i].date() != df.index[i-1].date())
        
        prev_confirmed = True
        
        if not bfvg1_aligned.iloc[i] and not sfvg1_aligned.iloc[i]:
            continue
        
        is_new_4h_curr = is_new_4h_aligned.iloc[i]
        
        if is_new_4h_curr and bfvg1_aligned.iloc[i] and lastFVG == -1:
            entry_price = close_4h.iloc[close_4h.index.get_loc(get_4h_idx(df.index[i]))] if get_4h_idx(df.index[i]) in close_4h.index else df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df.index[i]),
                'entry_time': datetime.fromtimestamp(df.index[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
            lastFVG = 1
        elif is_new_4h_curr and sfvg1_aligned.iloc[i] and lastFVG == 1:
            entry_price = close_4h.iloc[close_4h.index.get_loc(get_4h_idx(df.index[i]))] if get_4h_idx(df.index[i]) in close_4h.index else df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df.index[i]),
                'entry_time': datetime.fromtimestamp(df.index[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
            lastFVG = -1
        elif is_new_4h_curr:
            if bfvg1_aligned.iloc[i]:
                lastFVG = 1
            elif sfvg1_aligned.iloc[i]:
                lastFVG = -1
    
    return entries