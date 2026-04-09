import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    n = len(df)
    entries = []
    trade_num = 1
    
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, min_periods=20, adjust=False).mean()
    
    sma_vol = volume.rolling(9).mean()
    volfilt = volume.shift(1) > sma_vol * 1.5
    atrfilt = (low - high.shift(2) > atr / 1.5) | (low.shift(2) - high > atr / 1.5)
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2.copy()
    locfilts = ~loc2
    
    bfvg = (low > high.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (high < low.shift(2)) & volfilt & atrfilt & locfilts
    
    ts = df['time']
    hour = pd.to_datetime(ts, unit='s', utc=True).dt.hour
    minute = pd.to_datetime(ts, unit='s', utc=True).dt.minute
    isWithinMorningWindow = (hour == 8) & (minute <= 45)
    isWithinAfternoonWindow = (hour == 15) & (minute <= 45)
    isWithinTimeWindow = isWithinMorningWindow | isWithinAfternoonWindow
    
    obUp = (close.shift(2) < open.shift(2)) & (close.shift(1) > open.shift(1)) & (close.shift(1) > high.shift(2))
    obDown = (close.shift(2) > open.shift(2)) & (close.shift(1) < open.shift(1)) & (close.shift(1) < low.shift(2))
    fvgUp = low > high.shift(2)
    fvgDown = high < low.shift(2)
    
    long_condition = obUp & fvgUp & isWithinTimeWindow & bfvg
    short_condition = obDown & fvgDown & isWithinTimeWindow & sfvg
    
    for i in range(n):
        if i < 2 or i >= n - 3:
            continue
        if pd.isna(obUp.iloc[i]) or pd.isna(obDown.iloc[i]):
            continue
        if pd.isna(fvgUp.iloc[i]) or pd.isna(fvgDown.iloc[i]):
            continue
        if pd.isna(bfvg.iloc[i]) or pd.isna(sfvg.iloc[i]):
            continue
        if pd.isna(isWithinTimeWindow.iloc[i]):
            continue
            
        if long_condition.iloc[i]:
            ts_val = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts_val,
                'entry_time': datetime.fromtimestamp(ts_val, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
            
        if short_condition.iloc[i]:
            ts_val = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts_val,
                'entry_time': datetime.fromtimestamp(ts_val, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries