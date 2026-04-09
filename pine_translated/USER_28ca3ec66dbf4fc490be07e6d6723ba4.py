import pandas as pd
import numpy as np
from datetime import datetime, timezone

def compute_wilder_atr(df, period=20):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['date'] = df['datetime'].dt.date
    
    # Time filter: 02-05 and 10-12 UTC
    is_valid_time = ((df['hour'] >= 2) & (df['hour'] < 5)) | ((df['hour'] >= 10) & (df['hour'] < 12))
    
    # FVG conditions
    df['high_2'] = df['high'].shift(2)
    df['low_2'] = df['low'].shift(2)
    bfvg = df['low'] > df['high_2']
    sfvg = df['high'] < df['low_2']
    
    # Filters (disabled by default in Pine Script, treating as enabled for entry logic)
    df['vol_sma9'] = df['volume'].shift(1).rolling(9).mean()
    volfilt = df['volume'].shift(1) > df['vol_sma9'] * 1.5
    atr = compute_wilder_atr(df, 20) / 1.5
    atrfilt_bull = df['low'] - df['high_2'] > atr
    atrfilt_bear = df['low_2'] - df['high'] > atr
    atrfilt = atrfilt_bull | atrfilt_bear
    loc = df['close'].rolling(54).mean()
    locfiltb = loc > loc.shift(1)
    locfilts = ~locfiltb
    
    # Combined FVG conditions
    bfvg_cond = bfvg & volfilt & atrfilt & locfiltb
    sfvg_cond = sfvg & volfilt & atrfilt & locfilts
    
    # OB conditions
    is_up = df['close'] > df['open']
    is_down = df['close'] < df['open']
    ob_up = is_down.shift(2) & is_up.shift(1) & (df['close'].shift(1) > df['high'].shift(2))
    ob_down = is_up.shift(2) & is_down.shift(1) & (df['close'].shift(1) < df['low'].shift(2))
    
    # Entry signals
    long_cond = bfvg_cond & ob_up & is_valid_time
    short_cond = sfvg_cond & ob_down & is_valid_time
    
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if long_cond.iloc[i]:
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
        elif short_cond.iloc[i]:
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
    return entries