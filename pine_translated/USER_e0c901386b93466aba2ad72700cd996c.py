import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['date'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.date
    daily = df.groupby('date').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum')
    ).reset_index()
    daily['time'] = pd.to_datetime(daily['date']).astype('int64') // 10**6
    
    daily_high = daily['high']
    daily_low = daily['low']
    daily_close = daily['close']
    daily_open = daily['open']
    daily_vol = daily['volume']
    
    daily_high1 = daily_high.shift(1)
    daily_low1 = daily_low.shift(1)
    daily_high2 = daily_high.shift(2)
    daily_low2 = daily_low.shift(2)
    
    vol_sma = daily_vol.rolling(9).mean()
    
    tr1 = daily_high - daily_low
    tr2 = (daily_close - daily_high).abs()
    tr3 = (daily_low - daily_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/20, adjust=False).mean()
    
    loc = daily_close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    atrfilt = (daily_low - daily_high2 > atr/1.5) | (daily_low2 - daily_high > atr/1.5)
    
    bfvg = (daily_low > daily_high2) & (daily_vol > vol_sma * 1.5) & atrfilt & locfiltb
    sfvg = (daily_high < daily_low2) & (daily_vol > vol_sma * 1.5) & atrfilt & locfilts
    
    is_bullish_sharp = bfvg & (daily['high'].shift(1) < daily_low2.shift(1))
    is_bearish_sharp = sfvg & (daily['low'].shift(1) > daily_high2.shift(1))
    
    entries = []
    last_fvg_dir = 0
    
    for i in range(2, len(daily)):
        bullsharp = is_bullish_sharp.iloc[i] if not pd.isna(is_bullish_sharp.iloc[i]) else False
        bearsharp = is_bearish_sharp.iloc[i] if not pd.isna(is_bearish_sharp.iloc[i]) else False
        
        if bullsharp and last_fvg_dir == -1:
            ts = daily['time'].iloc[i]
            entries.append({
                'trade_num': len(entries) + 1,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': daily['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': daily['close'].iloc[i],
                'raw_price_b': daily['close'].iloc[i]
            })
            last_fvg_dir = 1
        elif bearsharp and last_fvg_dir == 1:
            ts = daily['time'].iloc[i]
            entries.append({
                'trade_num': len(entries) + 1,
                'direction': 'short',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': daily['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': daily['close'].iloc[i],
                'raw_price_b': daily['close'].iloc[i]
            })
            last_fvg_dir = -1
        elif bfvg.iloc[i] if not pd.isna(bfvg.iloc[i]) else False:
            last_fvg_dir = 1
        elif sfvg.iloc[i] if not pd.isna(sfvg.iloc[i]) else False:
            last_fvg_dir = -1
    
    return entries