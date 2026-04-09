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
    
    # Time range filtering - London session windows
    # Window 1: 07:00-11:45 UTC (Europe/London)
    # Window 2: 13:00-14:45 UTC (Europe/London)
    df['ts_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['ts_dt'].dt.hour
    df['minute'] = df['ts_dt'].dt.minute
    
    isWithinWindow1 = ((df['hour'] == 7) & (df['minute'] >= 0)) | \
                      ((df['hour'] >= 8) & (df['hour'] <= 10)) | \
                      ((df['hour'] == 11) & (df['minute'] <= 45))
    isWithinWindow2 = ((df['hour'] >= 13) & (df['hour'] <= 14)) | \
                      ((df['hour'] == 12) & (df['minute'] >= 0)) | \
                      ((df['hour'] == 14) & (df['minute'] <= 45))
    in_trading_window = isWithinWindow1 | isWithinWindow2
    
    # 4H timeframe conversion
    # Determine 4H period boundaries
    df['ts_dt_local'] = df['ts_dt'].dt.tz_convert('Europe/London')
    df['4h_period'] = (df['ts_dt_local'].dt.floor('4h').astype('int64') // 10**9)
    
    # Detect new 4H candle
    df['is_new_4h'] = df['4h_period'].diff().fillna(1) != 0
    df['is_new_4h'] = df['is_new_4h'].astype(bool)
    
    # Aggregate to 4H OHLCV
    agg_dict = {
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'open': 'first',
        'volume': 'sum',
        'time': 'first'
    }
    resampler_4h = df.resample('4h', on='ts_dt_local').agg(agg_dict)
    resampler_4h = resampler_4h.dropna(subset=['close'])
    resampler_4h = resampler_4h.reset_index()
    
    if len(resampler_4h) < 3:
        return results
    
    # 4H indicators
    high_4h = resampler_4h['high']
    low_4h = resampler_4h['low']
    close_4h = resampler_4h['close']
    volume_4h = resampler_4h['volume']
    
    # Volume filter
    vol_sma_4h = volume_4h.rolling(9).mean()
    volfilt = volume_4h.shift(1) > vol_sma_4h.shift(1) * 1.5
    
    # ATR filter (Wilder RSI method for ATR)
    high_low = high_4h - low_4h
    high_close = np.abs(high_4h - close_4h.shift(1))
    low_close = np.abs(low_4h - close_4h.shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_length = 20
    atr_4h = tr.ewm(alpha=1/atr_length, adjust=False).mean() / 1.5
    
    atrfilt = ((low_4h - high_4h.shift(2) > atr_4h) | 
               (low_4h.shift(2) - high_4h > atr_4h))
    
    # Trend filter (54 SMA on 4H close)
    loc1 = close_4h.rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    locfiltb = loc21
    locfilts = ~loc21
    
    # Bullish and Bearish FVG detection on 4H
    bfvg = (low_4h > high_4h.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (high_4h < low_4h.shift(2)) & volfilt & atrfilt & locfilts
    
    # Track last FVG type
    lastFVG = 0
    lastFVG_series = pd.Series(0, index=range(len(resampler_4h)))
    
    for i in range(len(resampler_4h)):
        if resampler_4h['is_new_4h'].iloc[i] if 'is_new_4h' in resampler_4h.columns else (i == 0):
            if bfvg.iloc[i] and lastFVG == -1:
                lastFVG_series.iloc[i] = 1
                lastFVG = 1
            elif sfvg.iloc[i] and lastFVG == 1:
                lastFVG_series.iloc[i] = -1
                lastFVG = -1
            elif bfvg.iloc[i]:
                lastFVG = 1
            elif sfvg.iloc[i]:
                lastFVG = -1
    
    # Detect bullish sharp turn (bfvg with lastFVG == -1)
    # Detect bearish sharp turn (sfvg with lastFVG == 1)
    bullish_sharp_turn = bfvg & (lastFVG_series.shift(1) == -1)
    bearish_sharp_turn = sfvg & (lastFVG_series.shift(1) == 1)
    
    # Session high/low calculation for breakout detection
    sess = df['ts_dt_local'].dt.strftime('%H:%M').apply(lambda x: '20:00' <= x <= '01:00')
    inside_sess = sess
    
    sess_high = df['high'].copy()
    sess_low = df['low'].copy()
    
    # Session breakout levels
    confirmed_high = df['high'].rolling(50).max().shift(1)
    confirmed_low = df['low'].rolling(50).min().shift(1)
    
    # Valid timeframe check (<=45min)
    valid_tf = True
    
    # Breakout conditions
    bullish_breakout = (df['close'] > confirmed_high) & valid_tf
    bearish_breakout = (df['close'] < confirmed_low) & valid_tf
    
    # Map 4H signals back to original timeframe
    # Create a mapping from 4H timestamp to signal
    resampler_4h['bullish_sharp_turn'] = bullish_sharp_turn.values
    resampler_4h['bearish_sharp_turn'] = bearish_sharp_turn.values
    
    df_merged = df.merge(
        resampler_4h[['ts_dt_local', 'bullish_sharp_turn', 'bearish_sharp_turn', 'close_4h']].rename(
            columns={'ts_dt_local': 'ts_dt_local', 'close_4h': 'close_4h_val'}
        ),
        on='ts_dt_local',
        how='left'
    )
    
    # Fill forward the 4H signals within each 4H period
    df_merged['bullish_sharp_turn'] = df_merged['bullish_sharp_turn'].fillna(method='ffill').fillna(False)
    df_merged['bearish_sharp_turn'] = df_merged['bearish_sharp_turn'].fillna(method='ffill').fillna(False)
    
    # Entry conditions
    long_condition = (df_merged['bullish_sharp_turn'] | bullish_breakout) & in_trading_window
    short_condition = (df_merged['bearish_sharp_turn'] | bearish_breakout) & in_trading_window
    
    # Generate entries
    for i in range(len(df)):
        if pd.isna(df['close'].iloc[i]):
            continue
            
        entry_price = df['close'].iloc[i]
        entry_ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        
        if long_condition.iloc[i]:
            results.append({
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
            
        if short_condition.iloc[i]:
            results.append({
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
    
    return results