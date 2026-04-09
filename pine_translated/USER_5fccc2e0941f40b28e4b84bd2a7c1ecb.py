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
    fastLength = 50
    slowLength = 200
    atrPeriod = 14
    start_hour = 7
    end_hour = 10
    end_minute = 59

    df = df.copy()
    df['time_dt'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    
    # Adjust for UTC+1 timezone
    df['time_utc1'] = df['time_dt'] + pd.Timedelta(hours=1)
    df['hour'] = df['time_utc1'].dt.hour
    df['minute'] = df['time_utc1'].dt.minute
    
    # Trading window logic
    in_trading_window = ((df['hour'] >= start_hour) & (df['hour'] <= end_hour)) & ~((df['hour'] == end_hour) & (df['minute'] > end_minute))
    
    # EMAs
    df['ema_fast'] = df['close'].ewm(span=fastLength, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slowLength, adjust=False).mean()
    
    # ATR (Wilder)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/atrPeriod, adjust=False).mean()
    
    # Detect new day
    df['day'] = df['time_dt'].dt.date
    df['isNewDay'] = df['day'] != df['day'].shift(1)
    
    # Previous day high/low (shift by 1 day lookback, simplified)
    df['prevDayHigh'] = df['high'].rolling(window=1440, min_periods=1).max().shift(1)
    df['prevDayLow'] = df['low'].rolling(window=1440, min_periods=1).min().shift(1)
    
    # Detect high/low sweeps
    df['sweepPDH'] = (df['close'] > df['prevDayHigh']) & (df['close'].shift(1) <= df['prevDayHigh'])
    df['sweepPDL'] = (df['close'] < df['prevDayLow']) & (df['close'].shift(1) >= df['prevDayLow'])
    
    # Reset sweep flags at new day
    df.loc[df['isNewDay'], 'sweepPDH'] = False
    df.loc[df['isNewDay'], 'sweepPDL'] = False
    
    # Detect FVG (Fair Value Gap) - bullish and bearish
    df['bullFVG'] = (df['low'] > df['high'].shift(2)) & (df['close'] > df['low'].shift(1))
    df['bearFVG'] = (df['high'] < df['low'].shift(2)) & (df['close'] < df['high'].shift(1))
    
    # EMA crossover signals
    df['ema_cross_up'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
    df['ema_cross_down'] = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
    
    # Long entry conditions: EMA uptrend + in window + sweep of PDL + FVG confirmation
    long_condition = (
        df['ema_fast'].notna() & df['ema_slow'].notna() &
        (df['ema_fast'] > df['ema_slow']) &
        in_trading_window &
        df['sweepPDL'] &
        df['bullFVG']
    )
    
    # Short entry conditions: EMA downtrend + in window + sweep of PDH + FVG confirmation
    short_condition = (
        df['ema_fast'].notna() & df['ema_slow'].notna() &
        (df['ema_fast'] < df['ema_slow']) &
        in_trading_window &
        df['sweepPDH'] &
        df['bearFVG']
    )
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(df['ema_fast'].iloc[i]) or pd.isna(df['ema_slow'].iloc[i]):
            continue
            
        if long_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
            
        if short_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries