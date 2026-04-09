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
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['datetime'].dt.date
    
    daily_open = df.groupby('date')['open'].transform('first')
    df['daily_open'] = daily_open
    df['isDailyGreen'] = df['close'] > df['daily_open']
    df['isDailyRed'] = df['close'] < df['daily_open']
    df['bullishAllowed'] = df['isDailyRed']
    df['bearishAllowed'] = df['isDailyGreen']
    
    prev_day_open = df.groupby('date')['high'].transform('first')
    df['newDay'] = df['date'] != df['date'].shift(1).fillna(df['date'])
    
    pdl = df.groupby('date')['low'].transform('first').shift(1)
    pdh = df.groupby('date')['high'].transform('first').shift(1)
    
    df['sweptLow'] = df['low'] < pdl
    df['sweptHigh'] = df['high'] > pdh
    df['brokeHigh'] = df['close'] > pdh
    df['brokeLow'] = df['close'] < pdl
    
    bias = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if df['newDay'].iloc[i]:
            bias.iloc[i] = 0
        elif df['sweptLow'].iloc[i] and df['brokeHigh'].iloc[i]:
            bias.iloc[i] = 1
        elif df['sweptHigh'].iloc[i] and df['brokeLow'].iloc[i]:
            bias.iloc[i] = -1
        elif df['low'].iloc[i] < pdl.iloc[i]:
            bias.iloc[i] = -1
        elif df['high'].iloc[i] > pdh.iloc[i]:
            bias.iloc[i] = 1
        else:
            bias.iloc[i] = bias.iloc[i-1]
    
    df['bias'] = bias
    
    bullfvghigh1 = (df['low'] > df['high'].shift(2)) & (df['low'] > df['high'].shift(2))
    bearfvglow1 = (df['high'] < df['low'].shift(2)) & (df['high'] < df['low'].shift(2))
    
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def wilder_atr(high, low, close, length):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        return atr
    
    rsi_4h = wilder_rsi(df['close'], 14)
    atr_4h = wilder_atr(df['high'], df['low'], df['close'], 14)
    
    bulltap = pd.Series(0, index=df.index)
    beartap = pd.Series(0, index=df.index)
    for i in range(3, len(df)):
        if bullfvghigh1.iloc[i]:
            bulltap.iloc[i] = bulltap.iloc[i-1] + 1
            beartap.iloc[i] = 0
        elif bearfvglow1.iloc[i]:
            beartap.iloc[i] = beartap.iloc[i-1] + 1
            bulltap.iloc[i] = 0
        else:
            bulltap.iloc[i] = 0
            beartap.iloc[i] = 0
    
    df['bulltap'] = bulltap
    df['beartap'] = beartap
    df['bullfvghigh1'] = bullfvghigh1
    df['bearfvglow1'] = bearfvglow1
    
    df['long_condition'] = (df['bias'] == 1) & (df['bulltap'] >= 3) & (df['bullfvghigh1']) & (df['bullishAllowed'])
    df['short_condition'] = (df['bias'] == -1) & (df['beartap'] >= 3) & (df['bearfvglow1']) & (df['bearishAllowed'])
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if df['long_condition'].iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif df['short_condition'].iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries