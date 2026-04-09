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
    
    df = df.copy().reset_index(drop=True)
    
    # Helper functions for conditions
    def is_up(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]
    
    def is_down(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]
    
    def is_ob_up(idx):
        return is_down(idx + 1) and is_up(idx) and df['close'].iloc[idx] > df['high'].iloc[idx + 1]
    
    def is_ob_down(idx):
        return is_up(idx + 1) and is_down(idx) and df['close'].iloc[idx] < df['low'].iloc[idx + 1]
    
    def is_fvg_up(idx):
        return df['low'].iloc[idx] > df['high'].iloc[idx + 2]
    
    def is_fvg_down(idx):
        return df['high'].iloc[idx] < df['low'].iloc[idx + 2]
    
    # Time window conditions (London time: 8:00-9:45 and 15:00-16:45)
    def in_trading_window(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        # Morning window: 8:00 to 9:45
        in_morning = (hour == 8 and minute >= 0) or (hour == 9 and minute <= 45)
        # Afternoon window: 15:00 to 16:45
        in_afternoon = (hour == 15 and minute >= 0) or (hour == 16 and minute <= 45)
        return in_morning or in_afternoon
    
    # Calculate filters
    # volfilt: volume[1] > ta.sma(volume, 9) * 1.5
    df['vol_sma9'] = df['volume'].rolling(9).mean()
    df['volfilt'] = df['volume'].shift(1) > df['vol_sma9'] * 1.5
    
    # ATR filter: ta.atr(20) / 1.5
    def calculate_wilder_atr(df, period=20):
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    df['atr'] = calculate_wilder_atr(df, 20)
    df['atr_adj'] = df['atr'] / 1.5
    df['atrfilt'] = ((df['low'] - df['high'].shift(2) > df['atr_adj']) | (df['low'].shift(2) - df['high'] > df['atr_adj']))
    
    # Trend filter: ta.sma(close, 54)
    df['loc'] = df['close'].rolling(54).mean()
    df['loc2'] = df['loc'] > df['loc'].shift(1)
    df['locfiltb'] = df['loc2']
    df['locfilts'] = ~df['loc2']
    
    # FVG conditions
    df['bfvg'] = (df['low'] > df['high'].shift(2)) & df['volfilt'] & df['atrfilt'] & df['locfiltb']
    df['sfvg'] = (df['high'] < df['low'].shift(2)) & df['volfilt'] & df['atrfilt'] & df['locfilts']
    
    # OB and FVG flags
    df['obUp'] = df.index.map(lambda i: is_ob_up(1) if i >= 2 else False)
    df['obDown'] = df.index.map(lambda i: is_ob_down(1) if i >= 2 else False)
    df['fvgUp'] = df.index.map(lambda i: is_fvg_up(0) if i >= 2 else False)
    df['fvgDown'] = df.index.map(lambda i: is_fvg_down(0) if i >= 2 else False)
    
    # Time window
    df['in_time_window'] = df['time'].apply(in_trading_window)
    
    # Entry conditions
    df['long_entry'] = df['obUp'] & df['fvgUp'] & df['in_time_window'] & df['bfvg']
    df['short_entry'] = df['obDown'] & df['fvgDown'] & df['in_time_window'] & df['sfvg']
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        direction = None
        if row['long_entry']:
            direction = 'long'
        elif row['short_entry']:
            direction = 'short'
        
        if direction:
            ts = int(row['time'])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = row['close']
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
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