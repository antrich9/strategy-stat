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
    
    df['datetime'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    
    in_trading_window_1 = ((df['hour'] >= 7) & ((df['hour'] < 10) | ((df['hour'] == 10) & (df['minute'] <= 59))))
    in_trading_window_2 = ((df['hour'] >= 15) & ((df['hour'] < 16) | ((df['hour'] == 16) & (df['minute'] <= 59))))
    in_trading_window = in_trading_window_1 | in_trading_window_2
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.ewm(alpha=1.0/20, adjust=False).mean()
    atr_val = atr / 1.5
    
    vol_sma = df['volume'].rolling(9).mean()
    vol_filt = df['volume'].shift(1) > vol_sma * 1.5
    
    loc = df['close'].ewm(span=54, adjust=False).mean()
    loc2 = loc > loc.shift(1)
    loc_filt_b = loc2
    loc_filt_s = ~loc2
    
    fvg_up = (df['low'] > df['high'].shift(2)) & vol_filt & ((df['low'].shift(2) - df['high'] > atr_val) | (df['low'] - df['high'].shift(2) > atr_val)) & loc_filt_b
    fvg_down = (df['high'] < df['low'].shift(2)) & vol_filt & ((df['high'].shift(2) - df['low'] > atr_val) | (df['high'] - df['low'].shift(2) > atr_val)) & loc_filt_s
    
    is_up = df['close'] > df['open']
    is_down = df['close'] < df['open']
    
    ob_up = is_down.shift(1) & is_up & (df['close'] > df['high'].shift(1))
    ob_down = is_up.shift(1) & is_down & (df['close'] < df['low'].shift(1))
    
    conditions = {
        'long': fvg_up & ob_up.shift(1) & in_trading_window,
        'short': fvg_down & ob_down.shift(1) & in_trading_window
    }
    
    entries = []
    trade_num = 1
    
    for i in range(2, len(df)):
        if pd.isna(fvg_up.iloc[i]) or pd.isna(ob_up.iloc[i]) or pd.isna(in_trading_window.iloc[i]):
            continue
        
        for direction in ['long', 'short']:
            if conditions[direction].iloc[i]:
                ts = int(df['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': direction,
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts/1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                })
                trade_num += 1
    
    return entries