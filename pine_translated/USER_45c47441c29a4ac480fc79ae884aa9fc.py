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
    if len(df) < 3:
        return []
    
    df = df.copy()
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('Europe/London')
    
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    
    def in_window(start_h, start_m, end_h, end_m):
        start = df['hour'] * 60 + df['minute'] >= start_h * 60 + start_m
        end = df['hour'] * 60 + df['minute'] <= end_h * 60 + end_m
        return start & end
    
    isWithinMorningWindow = in_window(8, 0, 9, 45)
    isWithinAfternoonWindow = in_window(15, 0, 16, 45)
    in_trading_window = isWithinMorningWindow | isWithinAfternoonWindow
    
    df['high_lag2'] = df['high'].shift(2)
    df['low_lag2'] = df['low'].shift(2)
    
    def calculate_wilder_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['rsi'] = calculate_wilder_rsi(df['close'], 14)
    
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = df['tr'].ewm(alpha=1/20, adjust=False).mean()
    atr = df['atr'] / 1.5
    
    df['sma54'] = df['close'].rolling(54).mean()
    df['sma54_lag1'] = df['sma54'].shift(1)
    df['loc2'] = df['sma54'] > df['sma54_lag1']
    
    df['vol_sma9'] = df['volume'].rolling(9).mean()
    df['vol_lag1'] = df['volume'].shift(1)
    vol_filter = df['vol_lag1'] > df['vol_sma9'] * 1.5
    
    df['atr_cond'] = (df['low'] - df['high_lag2'] > atr) | (df['low_lag2'] - df['high'] > atr)
    atrfilt = df['atr_cond']
    
    locfiltb = df['loc2']
    locfilts = ~df['loc2']
    
    df['bfvg'] = (df['low'] > df['high_lag2']) & vol_filter & atrfilt & locfiltb
    df['sfvg'] = (df['high'] < df['low_lag2']) & vol_filter & atrfilt & locfilts
    
    entries = []
    trade_num = 1
    
    for i in range(2, len(df)):
        if df['bfvg'].iloc[i] and in_trading_window.iloc[i]:
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
        elif df['sfvg'].iloc[i] and in_trading_window.iloc[i]:
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