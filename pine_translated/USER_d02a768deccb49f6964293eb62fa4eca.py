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
    from datetime import timezone
    
    # Convert timestamp to datetime for timezone operations
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Resample to daily to get previous day's high and low
    df_daily = df.set_index('datetime')
    
    # Get daily OHLC
    daily_ohlc = df_daily.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    # Previous day's high and low (shift by 1 to get previous complete day)
    prev_day_high = daily_ohlc['high'].shift(1)
    prev_day_low = daily_ohlc['low'].shift(1)
    
    # Merge back to intraday data
    df_daily['prev_day_high'] = prev_day_high
    df_daily['prev_day_low'] = prev_day_low
    df = df_daily.reset_index()
    
    # Filter for session times (8:00-11:00 EST)
    est_offset = pd.Timedelta(hours=5)
    df['datetime_est'] = df['datetime'] - est_offset
    df['hour'] = df['datetime_est'].dt.hour
    df['minute'] = df['datetime_est'].dt.minute
    
    session_mask = ((df['hour'] == 8) | ((df['hour'] >= 9) & (df['hour'] < 11))) & \
                   ~((df['hour'] == 10) & (df['minute'] > 59))
    df['is_session'] = session_mask
    
    # Body size conditions
    body_size_pct_c1 = 70
    body_size_pct_c3 = 70
    
    # Calculate body and range for pattern detection
    df['body_0'] = abs(df['close'] - df['open'])
    df['range_0'] = df['high'] - df['low']
    df['body_1'] = abs(df['close'].shift(1) - df['open'].shift(1))
    df['range_1'] = df['high'].shift(1) - df['low'].shift(1)
    df['body_2'] = abs(df['close'].shift(2) - df['open'].shift(2))
    df['range_2'] = df['high'].shift(2) - df['low'].shift(2)
    
    # Body size requirements for pattern
    body_req_c1 = df['body_2'] >= (df['range_2'] * body_size_pct_c1 / 100)
    body_req_c3 = df['body_0'] >= (df['range_0'] * body_size_pct_c3 / 100)
    
    # Bullish pattern conditions
    bull_cond = ((df['low'].shift(2) < df['prev_day_low']) | (df['low'].shift(1) < df['prev_day_low'])) & \
                (df['low'].shift(1) < df['low'].shift(2)) & \
                (df['high'].shift(1) < df['high'].shift(2)) & \
                (df['low'] > df['low'].shift(1)) & \
                (df['close'] > df['high'].shift(1))
    
    valid_bullish = bull_cond & body_req_c3 & body_req_c1
    
    # Bearish pattern conditions
    bear_cond = ((df['high'].shift(2) > df['prev_day_high']) | (df['high'].shift(1) > df['prev_day_high'])) & \
                (df['high'].shift(1) > df['high'].shift(2)) & \
                (df['low'].shift(1) > df['low'].shift(2)) & \
                (df['high'] < df['high'].shift(1)) & \
                (df['close'] < df['low'].shift(1))
    
    valid_bearish = bear_cond & body_req_c3 & body_req_c1
    
    # Determine pattern type for each row
    df['pattern_type'] = 0
    df.loc[valid_bullish, 'pattern_type'] = 1  # bullish
    df.loc[valid_bearish, 'pattern_type'] = -1  # bearish
    
    # Detect PDH and PDL crosses within session
    df['close_above_pdh'] = df['close'] > df['prev_day_high']
    df['close_below_pdl'] = df['close'] < df['prev_day_low']
    df['close_prev_above_pdh'] = df['close'].shift(1) > df['prev_day_high'].shift(1)
    df['close_prev_below_pdl'] = df['close'].shift(1) < df['prev_day_low'].shift(1)
    
    pdh_cross_up = df['close_above_pdh'] & ~df['close_prev_above_pdh']
    pdl_cross_down = df['close_below_pdl'] & ~df['close_prev_below_pdl']
    
    # Track when PDH/PDL gets broken in each session window
    df['session_day'] = df['datetime_est'].dt.date
    df['pdh_broken'] = df.groupby('session_day')['close_above_pdh'].cummax() & df['is_session']
    df['pdl_broken'] = df.groupby('session_day')['close_below_pdl'].cummax() & df['is_session']
    
    # Identify entry signals
    buy_signal = df['is_session'] & df['pdl_broken'] & (df['pattern_type'] == 1)
    sell_signal = df['is_session'] & df['pdh_broken'] & (df['pattern_type'] == -1)
    
    # Build the results list
    entries = []
    trade_num = 1
    
    for idx, row in df.iterrows():
        if buy_signal.iloc[idx]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(row['time']),
                'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                'entry_price_guess': row['close'],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': row['close'],
                'raw_price_b': row['close']
            })
            trade_num += 1
        elif sell_signal.iloc[idx]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(row['time']),
                'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                'entry_price_guess': row['close'],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': row['close'],
                'raw_price_b': row['close']
            })
            trade_num += 1
    
    return entries