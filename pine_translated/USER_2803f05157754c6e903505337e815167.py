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
    
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def wilder_atr(high, low, close, length):
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/length, adjust=False).mean()
        return atr
    
    df = df.copy()
    
    ts_utc = pd.to_datetime(df['time'], unit='s', utc=True)
    ts_london = ts_utc.dt.tz_convert('Europe/London')
    df['hour'] = ts_london.dt.hour
    df['minute'] = ts_london.dt.minute
    df['dayofweek'] = ts_london.dt.dayofweek
    
    morning_start = (df['hour'] == 8) & (df['minute'] >= 0)
    morning_end = (df['hour'] == 9) & (df['minute'] <= 55)
    isWithinMorningWindow = morning_start | morning_end
    
    afternoon_start = (df['hour'] == 14) & (df['minute'] >= 0)
    afternoon_end = (df['hour'] == 16) & (df['minute'] <= 55)
    isWithinAfternoonWindow = afternoon_start | afternoon_end
    
    isFridayMorningWindow = (df['dayofweek'] == 4) & isWithinMorningWindow
    
    ema200 = df['close'].ewm(span=200, adjust=False).mean()
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    
    lowest_14 = df['low'].rolling(14).min()
    highest_14 = df['high'].rolling(14).max()
    stoch = 100 * (df['close'] - lowest_14) / (highest_14 - lowest_14)
    kdj_k = stoch.rolling(3).mean()
    kdj_d = kdj_k.rolling(3).mean()
    kdj_j = 3 * kdj_k - 2 * kdj_d
    
    longCondition = (kdj_k > kdj_d) & (kdj_k.shift(1) <= kdj_d.shift(1)) & (df['close'] > ema200) & (df['close'] > ema50) & (df['close'] > ema20)
    shortCondition = (kdj_k < kdj_d) & (kdj_k.shift(1) >= kdj_d.shift(1)) & (df['close'] < ema200) & (df['close'] < ema50) & (df['close'] < ema20)
    
    in_trading_window = isWithinMorningWindow | isWithinAfternoonWindow
    
    entries = []
    trade_num = 1
    in_position = False
    
    for i in range(len(df)):
        if i < 5:
            continue
        
        if pd.isna(ema200.iloc[i]) or pd.isna(ema50.iloc[i]) or pd.isna(ema20.iloc[i]) or pd.isna(kdj_k.iloc[i]) or pd.isna(kdj_d.iloc[i]):
            continue
        
        if in_position:
            continue
        
        if not in_trading_window.iloc[i]:
            continue
        
        if isFridayMorningWindow.iloc[i]:
            continue
        
        entry_ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        
        if longCondition.iloc[i]:
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            in_position = True
        elif shortCondition.iloc[i]:
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            in_position = True
    
    return entries