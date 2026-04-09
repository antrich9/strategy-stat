import pandas as pd
import numpy as np
from datetime import datetime, timezone

def calc_atr(df, period=14):
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/period, adjust=False).mean()
    return atr

def calc_supertrend(df, period=10, multiplier=3):
    atr = calc_atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(0, index=df.index, dtype=int)
    
    supertrend.iloc[0] = lower_band.iloc[0]
    direction.iloc[0] = 1
    
    for i in range(1, len(df)):
        prev_close = df['close'].iloc[i-1]
        curr_close = df['close'].iloc[i]
        prev_st = supertrend.iloc[i-1]
        prev_dir = direction.iloc[i-1]
        
        upper = upper_band.iloc[i]
        lower = lower_band.iloc[i]
        
        if prev_dir == 1:
            if curr_close < lower:
                direction.iloc[i] = -1
                supertrend.iloc[i] = upper
            else:
                direction.iloc[i] = 1
                supertrend.iloc[i] = min(lower, prev_st) if prev_st > lower else lower
        else:
            if curr_close > upper:
                direction.iloc[i] = 1
                supertrend.iloc[i] = lower
            else:
                direction.iloc[i] = -1
                supertrend.iloc[i] = max(upper, prev_st) if prev_st < upper else upper
    
    return supertrend, direction

def calc_wilder_ma(series, period):
    return series.ewm(alpha=1.0/period, adjust=False).mean()

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    
    in_trading_window = (df['hour'] >= 7) & ((df['hour'] < 10) | ((df['hour'] == 10) & (df['minute'] <= 59)))
    
    ema_short = df['close'].ewm(span=8, adjust=False).mean()
    ema_long = df['close'].ewm(span=21, adjust=False).mean()
    
    ema_short_prev = ema_short.shift(1)
    ema_long_prev = ema_long.shift(1)
    ema_bullish = (ema_short > ema_long) & (ema_short_prev <= ema_long_prev)
    ema_bearish = (ema_short < ema_long) & (ema_short_prev >= ema_long_prev)
    
    supertrend_vals, supertrend_dir = calc_supertrend(df, period=10, multiplier=3)
    is_supertrend_bullish = supertrend_dir == 1
    is_supertrend_bearish = supertrend_dir == -1
    
    prev_close = df['close'].shift(1)
    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - prev_close).abs()
    low_close_prev = (df['low'] - prev_close).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    
    high_diff = df['high'] - df['high'].shift(1)
    low_diff = df['low'].shift(1) - df['low']
    
    for i in range(1, len(df)):
        if high_diff.iloc[i] > low_diff.iloc[i]:
            plus_dm.iloc[i] = max(high_diff.iloc[i], 0) if high_diff.iloc[i] > 0 else 0
        if low_diff.iloc[i] > high_diff.iloc[i]:
            minus_dm.iloc[i] = max(low_diff.iloc[i], 0) if low_diff.iloc[i] > 0 else 0
    
    smoothed_tr = calc_wilder_ma(tr, 14)
    smoothed_plus_dm = calc_wilder_ma(plus_dm, 14)
    smoothed_minus_dm = calc_wilder_ma(minus_dm, 14)
    
    plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
    minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = calc_wilder_ma(dx, 14)
    
    is_strong_trend = adx > 20
    
    long_condition = ema_bullish & is_supertrend_bullish & in_trading_window & is_strong_trend
    short_condition = ema_bearish & is_supertrend_bearish & in_trading_window & is_strong_trend
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i == 0:
            continue
        
        if long_condition.iloc[i]:
            entry_price = df['close'].iloc[i]
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            entry_price = df['close'].iloc[i]
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
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