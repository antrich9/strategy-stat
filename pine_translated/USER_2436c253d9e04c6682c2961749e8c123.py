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
    
    # Wilder EMA function (same as ta.rma in Pine)
    def wilder_ema(series, length):
        alpha = 1.0 / length
        return series.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    
    # Calculate True Range
    high = df['high']
    low = df['low']
    close = df['close']
    
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()
    
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    
    valid_up = (up_move > down_move) & (up_move > 0)
    valid_down = (down_move > up_move) & (down_move > 0)
    
    plus_dm[valid_up] = up_move[valid_up]
    minus_dm[valid_down] = down_move[valid_down]
    
    # Handle NaN at the beginning
    if plus_dm.iloc[0] == 0 and minus_dm.iloc[0] == 0:
        plus_dm.iloc[0] = np.nan
        minus_dm.iloc[0] = np.nan
    
    # Calculate ADX parameters
    adx_len = 14
    di_len = 14
    
    # Wilder smoothed True Range
    atr_wilder = wilder_ema(tr, di_len)
    
    # Fixnan for plus_dm and minus_dm before division
    plus_smoothed = wilder_ema(plus_dm.fillna(0), di_len)
    minus_smoothed = wilder_ema(minus_dm.fillna(0), di_len)
    
    # Avoid division by zero
    tr_safe = atr_wilder.replace(0, 1)
    
    plus_di = 100 * plus_smoothed / tr_safe
    minus_di = 100 * minus_smoothed / tr_safe
    
    sum_di = plus_di + minus_di
    sum_di_safe = sum_di.replace(0, 1)
    
    dx = 100 * wilder_ema((plus_di - minus_di).abs() / sum_di_safe, adx_len)
    
    adx_value = dx
    
    # ATR for stop loss calculation (ta.atr(14))
    atr_value = atr_wilder
    
    # Entry conditions
    long_condition = adx_value > 20
    short_condition = adx_value > 20
    
    entries = []
    trade_num = 1
    position_open = False
    current_direction = None
    
    for i in range(len(df)):
        if i == 0:
            continue
        
        if not position_open and not pd.isna(adx_value.iloc[i]):
            entry_price = close.iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            if long_condition.iloc[i]:
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
                position_open = True
                current_direction = 'long'
            elif short_condition.iloc[i]:
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
                position_open = True
                current_direction = 'short'
    
    return entries