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
    # Convert to 240T timeframe
    df_hf = df.copy()
    df_hf['time'] = pd.to_datetime(df_hf['time'], unit='s', utc=True)
    df_hf = df_hf.set_index('time')
    
    hf_agg = df_hf.resample('240T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    dailyHigh = hf_agg['high']
    dailyLow = hf_agg['low']
    dailyOpen = hf_agg['open']
    dailyClose = hf_agg['close']
    
    dailyHigh1 = dailyHigh.shift(1)
    dailyLow1 = dailyLow.shift(1)
    dailyHigh2 = dailyHigh.shift(2)
    dailyLow2 = dailyLow.shift(2)
    dailyHigh3 = dailyHigh.shift(3)
    dailyLow3 = dailyLow.shift(3)
    dailyHigh4 = dailyHigh.shift(4)
    dailyLow4 = dailyLow.shift(4)
    dailyClose1 = dailyClose.shift(1)
    
    # Filters (default to True since inputs are False in Pine Script)
    inp1, inp2, inp3 = False, False, False
    
    # Volume filter
    vol_sma = df['volume'].rolling(9).mean() * 1.5
    vol_filt = df['volume'].shift(1) > vol_sma
    
    # ATR filter (Wilder smoothing)
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atr2 = atr / 1.5
    
    # Reindex atr to hf
    daily_atr = atr.reindex(dailyHigh.index, method='ffill')
    
    atrfilt = ((dailyLow - dailyHigh2 > daily_atr / 1.5) | (dailyLow2 - dailyHigh > daily_atr / 1.5))
    atrfilt = atrfilt.reindex(df.index, method='ffill').fillna(False)
    
    # Trend filter using close SMA
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2 if inp3 else pd.Series(True, index=df.index)
    locfilts = ~loc2 if inp3 else pd.Series(True, index=df.index)
    
    # Bullish FVG
    bfvg = (dailyLow > dailyHigh2) & vol_filt & atrfilt & locfiltb
    # Bearish FVG
    sfvg = (dailyHigh < dailyLow2) & vol_filt & atrfilt & locfilts
    
    # Swing detection
    is_swing_high = (dailyHigh2 > dailyHigh1) & (dailyHigh2 > dailyHigh3) & (dailyHigh2 > dailyHigh4)
    is_swing_low = (dailyLow2 > dailyLow1) & (dailyLow2 > dailyLow3) & (dailyLow2 > dailyLow4)
    
    # Initialize lastSwingType
    last_swing_type = pd.Series('none', index=dailyHigh.index)
    last_swing_type[is_swing_high] = 'dailyHigh'
    last_swing_type[is_swing_low] = 'dailyLow'
    last_swing_type = last_swing_type.ffill().fillna('none')
    last_swing_type = last_swing_type.reindex(df.index, method='ffill').fillna('none')
    
    # Entry conditions
    bull_entry = bfvg & (last_swing_type == 'dailyLow')
    bear_entry = sfvg & (last_swing_type == 'dailyHigh')
    
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        ts = int(row['time'])
        price = float(row['close'])
        
        # Skip if indicators have NaN at this index
        if pd.isna(close.iloc[i]) or pd.isna(loc.iloc[i]) or pd.isna(atr.iloc[i]):
            continue
        hfi = df_hf.index.get_indexer([df['time'].iloc[i]], method='pad')[0]
        if hfi < 0 or pd.isna(dailyHigh.iloc[hfi]) or pd.isna(dailyLow.iloc[hfi]):
            continue
        
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        if bull_entry.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
        elif bear_entry.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
    
    return entries