import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    
    dt = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = dt.dt.hour
    df['minute'] = dt.dt.minute
    
    in_tw1 = (df['hour'] == 7) | ((df['hour'] >= 8) & (df['hour'] <= 9)) | ((df['hour'] == 10) & (df['minute'] <= 59))
    in_tw2 = (df['hour'] == 15) | ((df['hour'] == 16) & (df['minute'] <= 59))
    in_trading_window = in_tw1 | in_tw2
    
    sma_vol = df['volume'].rolling(9).mean()
    volfilt = df['volume'].shift(1) > sma_vol * 1.5
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atrfilt = (low - high.shift(2) > atr/1.5) | (low.shift(2) - high > atr/1.5)
    
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    isUp = close > df['open']
    isDown = close < df['open']
    
    isObUp = isDown.shift(2) & isUp & (close > high.shift(2))
    isObDown = isUp.shift(2) & isDown & (close < low.shift(2))
    
    fvgUp = low > high.shift(2)
    fvgDown = high < low.shift(2)
    
    obUp = isObUp.shift(1)
    obDown = isObDown.shift(1)
    
    bull_stack = obUp & fvgUp & in_trading_window
    bear_stack = obDown & fvgDown & in_trading_window
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if bull_stack.iloc[i]:
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
        elif bear_stack.iloc[i]:
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