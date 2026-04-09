import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df = df.sort_values('time').reset_index(drop=True)
    
    # Wilder ATR implementation
    def wilder_atr(df, length=14):
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr
    
    atr = wilder_atr(df, 20) / 1.5
    
    # Volume filter
    volfilt = df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5
    
    # ATR filter
    atrfilt = ((df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr))
    
    # Trend filter
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # FVG detection
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    # Higher timeframe EMA
    df['_ts'] = pd.to_datetime(df['time'], unit='s', utc=True)
    daily = df[df['_ts'].dt.time == pd.Timestamp('00:00:00').time()].copy()
    if len(daily) > 0:
        daily = daily.sort_values('time').reset_index(drop=True)
        daily['htf_ema'] = daily['close'].ewm(span=50, adjust=False).mean()
        merged = pd.merge_asof(df[['time']].sort_values('time'), 
                               daily[['time', 'htf_ema']].sort_values('time'), 
                               on='time', direction='backward')
        df['htf_ema'] = merged['htf_ema'].values
    else:
        df['htf_ema'] = np.nan
    
    # Swing detection
    main_bar_high = df['high'].shift(5)
    main_bar_low = df['low'].shift(5)
    is_swing_high = (df['high'].shift(4) < main_bar_high) & (df['high'].shift(6) < main_bar_high)
    is_swing_low = (df['low'].shift(4) > main_bar_low) & (df['low'].shift(6) > main_bar_low)
    
    # Entry conditions
    bullish_entry = bfvg & (df['lastFVG'].shift(1) == -1) & (df['htf_ema'] > df['htf_ema'].shift(1))
    bearish_entry = sfvg & (df['lastFVG'].shift(1) == 1) & (df['htf_ema'] < df['htf_ema'].shift(1))
    
    trade_num = 1
    lastFVG = 0
    entries = []
    
    for i in range(len(df)):
        if bullish_entry.iloc[i]:
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
            trade_num += 1
            lastFVG = 1
        elif bearish_entry.iloc[i]:
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
            trade_num += 1
            lastFVG = -1
        elif bfvg.iloc[i]:
            lastFVG = 1
        elif sfvg.iloc[i]:
            lastFVG = -1
    
    return entries