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
    results = []
    trade_num = 1
    
    # Timezone for Asia session
    tz_asia = timezone.utc  # Using UTC for simplicity
    
    # Asia session parameters
    asia_start_hour = 20
    asia_start_minute = 0
    asia_end_hour = 0
    asia_end_minute = 0
    
    # Calculate session boundaries
    df_session = df.copy()
    df_session['datetime'] = pd.to_datetime(df_session['time'], unit='s', utc=True)
    df_session['hour'] = df_session['datetime'].dt.hour
    df_session['minute'] = df_session['datetime'].dt.minute
    
    # Session detection: 20:00 to 00:00 (next day)
    def in_asia_session(row):
        hour = row['hour']
        minute = row['minute']
        if asia_start_hour <= 23:
            if asia_end_hour == 0:
                # Overnight session
                if hour >= asia_start_hour:
                    if hour == asia_start_hour:
                        return minute >= asia_start_minute
                    return True
                elif hour < asia_end_hour:
                    return True
                return False
            else:
                return (hour > asia_start_hour or (hour == asia_start_hour and minute >= asia_start_minute)) and \
                       (hour < asia_end_hour or (hour == asia_end_hour and minute < asia_end_minute))
        return False
    
    df_session['in_session'] = df_session.apply(in_asia_session, axis=1)
    
    # Session high/low tracking
    session_high = pd.Series([np.nan] * len(df_session), index=df_session.index)
    session_low = pd.Series([np.nan] * len(df_session), index=df_session.index)
    swept_high = pd.Series([False] * len(df_session), index=df_session.index)
    swept_low = pd.Series([False] * len(df_session), index=df_session.index)
    
    # Detect session start
    df_session['session_start_prev'] = (df_session['in_session']) & (~df_session['in_session'].shift(1, fill_value=False))
    
    prev_high = np.nan
    prev_low = np.nan
    prev_swept_high = False
    prev_swept_low = False
    
    for i in range(len(df_session)):
        if i > 0:
            prev_high = session_high.iloc[i-1] if not pd.isna(session_high.iloc[i-1]) else np.nan
            prev_low = session_low.iloc[i-1] if not pd.isna(session_low.iloc[i-1]) else np.nan
            prev_swept_high = swept_high.iloc[i-1]
            prev_swept_low = swept_low.iloc[i-1]
        
        if df_session['in_session'].iloc[i]:
            if pd.isna(prev_high):
                session_high.iloc[i] = df_session['high'].iloc[i]
            else:
                session_high.iloc[i] = max(prev_high, df_session['high'].iloc[i])
            
            if pd.isna(prev_low):
                session_low.iloc[i] = df_session['low'].iloc[i]
            else:
                session_low.iloc[i] = min(prev_low, df_session['low'].iloc[i])
        else:
            session_high.iloc[i] = prev_high
            session_low.iloc[i] = prev_low
        
        if df_session['session_start_prev'].iloc[i]:
            session_high.iloc[i] = np.nan
            session_low.iloc[i] = np.nan
            swept_high.iloc[i] = False
            swept_low.iloc[i] = False
            prev_swept_high = False
            prev_swept_low = False
        else:
            swept_high.iloc[i] = prev_swept_high
            swept_low.iloc[i] = prev_swept_low
        
        # Sweep detection
        if not df_session['in_session'].iloc[i] and not swept_high.iloc[i]:
            if not pd.isna(session_high.iloc[i]) and df_session['high'].iloc[i] > session_high.iloc[i]:
                swept_high.iloc[i] = True
        
        if not df_session['in_session'].iloc[i] and not swept_low.iloc[i]:
            if not pd.isna(session_low.iloc[i]) and df_session['low'].iloc[i] < session_low.iloc[i]:
                swept_low.iloc[i] = True
    
    # Sweep conditions
    df_session['sweep_high_now'] = (~df_session['in_session']) & (~swept_high) & (~session_high.isna()) & (df_session['high'] > session_high)
    df_session['sweep_low_now'] = (~df_session['in_session']) & (~swept_low) & (~session_low.isna()) & (df_session['low'] < session_low)
    
    # Both swept flag
    df_session['both_swept'] = swept_high & swept_low
    
    # Daily open from security (simulated)
    df_session['daily_open'] = df_session['open'].cummax()  # Simplified - actual impl would use security
    df_session['is_daily_green'] = df_session['close'] > df_session['daily_open']
    df_session['is_daily_red'] = df_session['close'] < df_session['daily_open']
    
    # Bias allowed conditions
    df_session['bullish_allowed'] = df_session['is_daily_red']
    df_session['bearish_allowed'] = df_session['is_daily_green']
    
    # Previous day high/low (simulated from rolling max/min)
    lookback = 1440  # Approximate bars per day
    df_session['pdh'] = df_session['high'].shift(1).rolling(min_periods=min(lookback, i), window=lookback, min_periods=1).max().shift(1)
    df_session['pdl'] = df_session['low'].shift(1).rolling(min_periods=min(lookback, i), window=lookback, min_periods=1).min().shift(1)
    df_session['pd_close'] = df_session['close'].shift(1).rolling(min_periods=min(lookback, i), window=lookback, min_periods=1).last().shift(1)
    
    # New day detection
    df_session['new_day'] = df_session['datetime'].dt.date != df_session['datetime'].dt.date.shift(1, fill_value=df_session['datetime'].iloc[0])
    
    # Bias tracking
    bias = pd.Series([0] * len(df_session), index=df_session.index)
    
    # Swept conditions
    df_session['swept_low'] = df_session['low'] < df_session['pdl']
    df_session['swept_high'] = df_session['high'] > df_session['pdh']
    df_session['broke_high'] = df_session['close'] > df_session['pdh']
    df_session['broke_low'] = df_session['close'] < df_session['pdl']
    
    for i in range(1, len(df_session)):
        if df_session['new_day'].iloc[i]:
            bias.iloc[i] = 0
        else:
            bias.iloc[i] = bias.iloc[i-1]
        
        if df_session['swept_low'].iloc[i] and df_session['broke_high'].iloc[i]:
            bias.iloc[i] = 1
        elif df_session['swept_high'].iloc[i] and df_session['broke_low'].iloc[i]:
            bias.iloc[i] = -1
        elif df_session['low'].iloc[i] < df_session['pdl'].iloc[i]:
            bias.iloc[i] = -1
        elif df_session['high'].iloc[i] > df_session['pdh'].iloc[i]:
            bias.iloc[i] = 1
    
    # 4H data (simulated using 4-bar rolling for 15m chart)
    df_session['is_new_4h'] = df_session['datetime'].dt.floor('4h') != df_session['datetime'].shift(1, fill_value=df_session['datetime'].iloc[0]).dt.floor('4h')
    
    # 4H OHLCV (simulated)
    df_session['high_4h'] = df_session['high'].rolling(16, min_periods=1).max()
    df_session['low_4h'] = df_session['low'].rolling(16, min_periods=1).min()
    df_session['close_4h'] = df_session['close']
    df_session['volume_4h'] = df_session['volume'].rolling(16, min_periods=1).sum()
    
    # Volume filter
    volfilt = df_session['volume_4h'].shift(1) > df_session['volume_4h'].shift(1).rolling(9, min_periods=1).mean() * 1.5
    
    # ATR filter
    high_diff = df_session['low_4h'] - df_session['high_4h'].shift(2)
    low_diff = df_session['low_4h'].shift(2) - df_session['high_4h']
    tr1 = df_session['high'] - df_session['low']
    tr2 = abs(df_session['high'] - df_session['close'].shift(1))
    tr3 = abs(df_session['low'] - df_session['close'].shift(1))
    df_session['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).ewm(span=20, adjust=False).mean()
    atr_4h = df_session['atr'].rolling(16, min_periods=1).max() / 1.5
    atrfilt = (high_diff > atr_4h) | (low_diff > atr_4h)
    
    # Trend filter
    loc1 = df_session['close_4h'].ewm(span=54, adjust=False).mean()
    locfiltb = loc1 > loc1.shift(1)
    locfilts = loc1 < loc1.shift(1)
    
    # FVG detection
    bfvg = (df_session['low_4h'] > df_session['high_4h'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (df_session['high_4h'] < df_session['low_4h'].shift(2)) & volfilt & atrfilt & locfilts
    
    # Entry conditions based on strategy logic
    long_entry_cond = df_session['sweep_low_now'] & df_session['bullish_allowed'] & (bias == 1) & bfvg & ~df_session['both_swept']
    short_entry_cond = df_session['sweep_high_now'] & df_session['bearish_allowed'] & (bias == -1) & sfvg & ~df_session['both_swept']
    
    df_session['long_entry'] = long_entry_cond
    df_session['short_entry'] = short_entry_cond
    
    # Generate entries
    for i in range(len(df_session)):
        if i < 2:
            continue
        
        entry_price = df_session['close'].iloc[i]
        ts = int(df_session['time'].iloc[i])
        
        if df_session['long_entry'].iloc[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=tz_asia).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
        elif df_session['short_entry'].iloc[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=tz_asia).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
    
    return results