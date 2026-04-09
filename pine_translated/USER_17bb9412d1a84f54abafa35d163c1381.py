import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

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
    
    # Wilder RSI / ATR implementation
    def calc_atr(high, low, close, length=14):
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(length).mean()
        for i in range(length, len(tr)):
            if not pd.isna(atr.iloc[i-1]):
                atr.iloc[i] = (atr.iloc[i-1] * (length - 1) + tr.iloc[i]) / length
        return atr
    
    def calc_ema(series, length):
        return series.ewm(span=length, adjust=False).mean()
    
    def calc_supertrend(high, low, close, period=10, multiplier=3.0):
        atr = calc_atr(high, low, close, period)
        hl2 = (high + low) / 2
        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr
        supertrend = pd.Series(np.nan, index=close.index)
        trend = pd.Series(0, index=close.index)
        
        for i in range(period, len(close)):
            if pd.isna(atr.iloc[i]):
                continue
            if close.iloc[i] > upper.iloc[i]:
                trend.iloc[i] = 1
            elif close.iloc[i] < lower.iloc[i]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1] if not pd.isna(trend.iloc[i-1]) else 0
            
            if trend.iloc[i] == 1:
                supertrend.iloc[i] = lower.iloc[i]
            elif trend.iloc[i] == -1:
                supertrend.iloc[i] = upper.iloc[i]
        
        return supertrend, trend
    
    ema_short = calc_ema(df['close'], 8)
    ema_long = calc_ema(df['close'], 21)
    
    ema_bullish = (ema_short > ema_long) & (ema_short.shift(1) <= ema_long.shift(1))
    ema_bearish = (ema_short < ema_long) & (ema_short.shift(1) >= ema_long.shift(1))
    
    _, supertrend_direction = calc_supertrend(df['high'], df['low'], df['close'], 10, 3.0)
    supertrend_bullish = supertrend_direction == 1
    supertrend_bearish = supertrend_direction == -1
    
    start_hour = 7
    end_hour = 10
    end_minute = 59
    
    in_window = pd.Series(False, index=df.index)
    for i in df.index:
        ts = df['time'].iloc[i]
        dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        month = dt.month
        day = dt.day
        dayofweek = dt.weekday()
        hour = dt.hour
        
        is_dst = (month >= 3 and (month == 3 and dayofweek == 6 and day >= 25 or month > 3)) and \
                 (month <= 10 and (month == 10 and dayofweek == 6 and day < 25 or month < 10))
        
        offset_hours = 2 if is_dst else 1
        adj_dt = dt + timedelta(hours=offset_hours)
        adj_hour = adj_dt.hour
        
        in_window.iloc[i] = (adj_hour >= start_hour and adj_hour <= end_hour) and not (adj_hour == end_hour and adj_dt.minute > end_minute)
    
    position = 'none'
    trade_num = 1
    entries = []
    
    for i in range(1, len(df)):
        if pd.isna(ema_bullish.iloc[i]) or pd.isna(supertrend_bullish.iloc[i]):
            continue
        
        if position == 'none':
            if ema_bullish.iloc[i] and supertrend_bullish.iloc[i] and in_window.iloc[i]:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000.0, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                })
                position = 'long'
                trade_num += 1
            elif ema_bearish.iloc[i] and supertrend_bearish.iloc[i] and in_window.iloc[i]:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000.0, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                })
                position = 'short'
                trade_num += 1
    
    return entries