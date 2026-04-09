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
    
    open_col = df['open']
    high_col = df['high']
    low_col = df['low']
    close_col = df['close']
    volume_col = df['volume']
    
    # OB conditions (using bar offset logic from Pine)
    is_up_0 = close_col > open_col
    is_down_0 = close_col < open_col
    is_up_1 = close_col.shift(1) > open_col.shift(1)
    is_down_1 = close_col.shift(1) < open_col.shift(1)
    
    ob_up = is_down_1.shift(1) & is_up_0 & (close_col > high_col.shift(1))
    ob_down = is_up_1.shift(1) & is_down_0 & (close_col < low_col.shift(1))
    
    fvg_up = low_col > high_col.shift(2)
    fvg_down = high_col < low_col.shift(2)
    
    # Time window logic
    timestamps = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in df['time']]
    hours = np.array([dt.hour for dt in timestamps])
    minutes = np.array([dt.minute for dt in timestamps])
    
    start_hour_1, end_hour_1 = 7, 10
    end_minute_1 = 59
    start_hour_2, end_hour_2 = 15, 16
    end_minute_2 = 59
    
    in_window_1 = (hours >= start_hour_1) & (hours <= end_hour_1) & ~((hours == end_hour_1) & (minutes > end_minute_1))
    in_window_2 = (hours >= start_hour_2) & (hours <= end_hour_2) & ~((hours == end_hour_2) & (minutes > end_minute_2))
    in_trading_window = in_window_1 | in_window_2
    
    # Filters
    vol_sma = volume_col.rolling(9).mean()
    volfilt = volume_col.shift(1) > vol_sma * 1.5
    
    atr = _wilder_atr(df, 20) / 1.5
    atrfilt = ((low_col - high_col.shift(2) > atr) | (low_col.shift(2) - high_col > atr))
    
    loc = close_col.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # Entry conditions
    bfvg = fvg_up & volfilt & atrfilt & locfiltb
    sfvg = fvg_down & volfilt & atrfilt & locfilts
    
    long_cond = bfvg & in_trading_window & ob_up
    short_cond = sfvg & in_trading_window & ob_down
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_cond.iloc[i]:
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close_col.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close_col.iloc[i],
                'raw_price_b': close_col.iloc[i]
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close_col.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close_col.iloc[i],
                'raw_price_b': close_col.iloc[i]
            })
            trade_num += 1
    
    return entries

def _wilder_atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr