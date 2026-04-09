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
    df['ts_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Create daily OHLCV
    daily = df.groupby(df['ts_dt'].dt.date).agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum')
    ).reset_index()
    daily.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    # Daily shifted values
    daily['dailyHigh11'] = daily['high']
    daily['dailyLow11'] = daily['low']
    daily['dailyHigh21'] = daily['high'].shift(1)
    daily['dailyLow21'] = daily['low'].shift(1)
    daily['dailyHigh22'] = daily['high'].shift(2)
    daily['dailyLow22'] = daily['low'].shift(2)
    
    # FVG conditions
    daily['bfvg11'] = daily['dailyLow11'] > daily['dailyHigh22']
    daily['sfvg11'] = daily['dailyHigh11'] < daily['dailyLow22']
    
    # Volume filter
    vol_sma9 = df['volume'].rolling(9).mean()
    daily['volfilt'] = (df['volume'].shift(1) > vol_sma9.shift(1) * 1.5).values
    
    # ATR filter (Wilder method for 20 periods)
    tr = np.maximum(df['high'] - df['low'], 
                    np.maximum(np.abs(df['high'] - df['close'].shift(1)),
                               np.abs(df['low'] - df['close'].shift(1))))
    atr_raw = tr.ewm(alpha=1/20, adjust=False).mean()
    daily_tr = df.groupby(df['ts_dt'].dt.date)['high'].max() - df.groupby(df['ts_dt'].dt.date)['low'].min()
    daily_tr.index = pd.to_datetime(daily_tr.index)
    daily_tr = daily_tr.reindex(daily['date']).values
    daily['atr211'] = atr_raw.groupby(df['ts_dt'].dt.date).mean().values / 1.5
    
    # ATR condition for filter
    daily['atr_filter_check'] = daily['dailyLow22'] - daily['dailyHigh22'] > daily['atr211']
    daily['atr_filter_check2'] = daily['dailyLow22'] - daily['dailyHigh11'] > daily['atr211']
    daily['atrfilt11'] = (daily['atr_filter_check']) | (daily['atr_filter_check2'])
    
    # Trend filter - 54-bar SMA direction
    df['sma54'] = df['close'].rolling(54).mean()
    df['sma54_prev'] = df['sma54'].shift(1)
    daily['loc211'] = df.groupby(df['ts_dt'].dt.date)['sma54'].last().values > \
                      df.groupby(df['ts_dt'].dt.date)['sma54_prev'].last().values
    
    daily['locfiltb11'] = daily['loc211']
    daily['locfilts11'] = ~daily['loc211']
    
    # Swing detection
    daily['is_swing_high'] = (daily['dailyHigh21'] < daily['dailyHigh22']) & \
                              (daily['dailyHigh11'].shift(3) < daily['dailyHigh22']) & \
                              (daily['dailyHigh11'].shift(4) < daily['dailyHigh22'])
    daily['is_swing_low'] = (daily['dailyLow21'] > daily['dailyLow22']) & \
                             (daily['dailyLow11'].shift(3) > daily['dailyLow22']) & \
                             (daily['dailyLow11'].shift(4) > daily['dailyLow22'])
    
    # Track last swing type
    last_swing_type = "none"
    last_swing_high = np.nan
    last_swing_low = np.nan
    
    for idx in daily.index:
        if daily.loc[idx, 'is_swing_high']:
            last_swing_high = daily.loc[idx, 'dailyHigh22']
            last_swing_type = "dailyHigh"
        if daily.loc[idx, 'is_swing_low']:
            last_swing_low = daily.loc[idx, 'dailyLow22']
            last_swing_type = "dailyLow"
        daily.loc[idx, 'lastSwingType'] = last_swing_type
    
    # Time window check (London time)
    df['hour'] = df['ts_dt'].dt.hour
    df['minute'] = df['ts_dt'].dt.minute
    morning_start = (df['hour'] == 8) & (df['minute'] >= 0)
    morning_end = (df['hour'] == 9) & (df['minute'] <= 55)
    is_morning = morning_start | morning_end
    
    afternoon_start = (df['hour'] == 14) & (df['minute'] >= 0)
    afternoon_end = (df['hour'] == 16) & (df['minute'] <= 55)
    is_afternoon = afternoon_start | afternoon_end
    
    df['in_time_window'] = is_morning | is_afternoon
    
    # Map daily conditions to intraday df
    df['date_dt'] = df['ts_dt'].dt.date
    daily_dict = daily.set_index('date').to_dict('index')
    
    df['bfvg11'] = df['date_dt'].map(lambda x: daily_dict.get(x, {}).get('bfvg11', False))
    df['sfvg11'] = df['date_dt'].map(lambda x: daily_dict.get(x, {}).get('sfvg11', False))
    df['lastSwingType'] = df['date_dt'].map(lambda x: daily_dict.get(x, {}).get('lastSwingType', 'none'))
    df['loc211'] = df['date_dt'].map(lambda x: daily_dict.get(x, {}).get('loc211', False))
    df['atrfilt11'] = df['date_dt'].map(lambda x: daily_dict.get(x, {}).get('atrfilt11', True))
    df['volfilt'] = df['date_dt'].map(lambda x: daily_dict.get(x, {}).get('volfilt', True))
    
    # Entry conditions
    long_condition = df['bfvg11'] & (df['lastSwingType'] == 'dailyLow') & df['in_time_window']
    short_condition = df['sfvg11'] & (df['lastSwingType'] == 'dailyHigh') & df['in_time_window']
    
    # Build entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        direction = None
        if long_condition.iloc[i]:
            direction = 'long'
        elif short_condition.iloc[i]:
            direction = 'short'
        
        if direction is not None:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
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