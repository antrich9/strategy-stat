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
    
    daily_high = df.groupby('date')['high'].max()
    daily_low = df.groupby('date')['low'].min()
    
    df['prev_day_high'] = df['date'].map(daily_high.shift(-1))
    df['prev_day_low'] = df['date'].map(daily_low.shift(-1))
    
    bullish_bar = 1
    bearish_bar = -1
    body_pct_c3 = 70
    body_pct_c1 = 70
    
    range_c3 = np.abs(df['high'] - df['low'])
    body_req_c3 = np.abs(df['close'] - df['open']) >= (range_c3 * body_pct_c3 / 100)
    
    range_c1 = np.abs(df['high'].shift(2) - df['low'].shift(2))
    body_req_c1 = np.abs(df['close'].shift(2) - df['open'].shift(2)) >= (range_c1 * body_pct_c1 / 100)
    
    bullish_condition = (
        ((df['low'].shift(2) < df['prev_day_low']) | (df['low'].shift(1) < df['prev_day_low'])) &
        (df['low'].shift(1) < df['low'].shift(2)) &
        (df['high'].shift(1) < df['high'].shift(2)) &
        (df['low'] > df['low'].shift(1)) &
        (df['close'] > df['high'].shift(1))
    )
    
    bearish_condition = (
        ((df['high'].shift(2) > df['prev_day_high']) | (df['high'].shift(1) > df['prev_day_high'])) &
        (df['high'].shift(1) > df['high'].shift(2)) &
        (df['low'].shift(1) > df['low'].shift(2)) &
        (df['high'] < df['high'].shift(1)) &
        (df['close'] < df['low'].shift(1))
    )
    
    valid_bullish = bullish_condition & body_req_c3 & body_req_c1
    valid_bearish = bearish_condition & body_req_c3 & body_req_c1
    
    df['pattern_type'] = np.where(valid_bullish, bullish_bar, np.where(valid_bearish, bearish_bar, 0))
    
    volfilt = df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5
    atr_val = (df['high'] - df['low']).ewm(span=20, adjust=False).mean() / 1.5
    atrfilt = ((df['low'] - df['high'].shift(2)) > atr_val) | ((df['low'].shift(2) - df['high']) > atr_val)
    loc = df['close'].ewm(span=54, adjust=False).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    bullish_entry = (df['pattern_type'] == bullish_bar) & bfvg
    bearish_entry = (df['pattern_type'] == bearish_bar) & sfvg
    
    trades = []
    trade_num = 1
    
    for i in range(len(df)):
        if bullish_entry.iloc[i]:
            entry_price = df['close'].iloc[i]
            trade = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            }
            trades.append(trade)
            trade_num += 1
        elif bearish_entry.iloc[i]:
            entry_price = df['close'].iloc[i]
            trade = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            }
            trades.append(trade)
            trade_num += 1
    
    return trades