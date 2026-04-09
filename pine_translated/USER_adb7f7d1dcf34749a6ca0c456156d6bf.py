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
    
    # Helper function for timestamp conversion
    def ts_to_dt(ts):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    
    # Extract datetime from timestamp
    df = df.copy()
    df['dt'] = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc))
    df['hour'] = df['dt'].apply(lambda x: x.hour)
    df['dayofmonth'] = df['dt'].apply(lambda x: x.day)
    df['month'] = df['dt'].apply(lambda x: x.month)
    df['year'] = df['dt'].apply(lambda x: x.year)
    
    # === Asian Session Detection ===
    asian_start_hour = 23
    asian_end_hour = 6
    
    in_asian_session = (asian_start_hour > asian_end_hour) & (
        (df['hour'] >= asian_start_hour) | (df['hour'] < asian_end_hour)
    )
    
    in_asian_session_prev = in_asian_session.shift(1).fillna(False).astype(bool)
    asian_session_ended = (~in_asian_session) & in_asian_session_prev
    
    # === Asian Session High/Low Calculation ===
    asian_high = pd.Series(index=df.index, dtype=float)
    asian_low = pd.Series(index=df.index, dtype=float)
    temp_high = pd.Series(index=df.index, dtype=float)
    temp_low = pd.Series(index=df.index, dtype=float)
    
    current_temp_high = np.nan
    current_temp_low = np.nan
    current_asian_high = np.nan
    current_asian_low = np.nan
    
    for i in range(len(df)):
        if in_asian_session.iloc[i]:
            if np.isnan(current_temp_high):
                current_temp_high = df['high'].iloc[i]
            else:
                current_temp_high = max(current_temp_high, df['high'].iloc[i])
            
            if np.isnan(current_temp_low):
                current_temp_low = df['low'].iloc[i]
            else:
                current_temp_low = min(current_temp_low, df['low'].iloc[i])
        
        if asian_session_ended.iloc[i]:
            asian_high.iloc[i] = current_temp_high
            asian_low.iloc[i] = current_temp_low
            current_temp_high = np.nan
            current_temp_low = np.nan
    
    asian_high = asian_high.ffill()
    asian_low = asian_low.ffill()
    
    # === Asia Sweep Detection ===
    swept_high = False
    swept_low = False
    
    sweep_high_now = pd.Series(False, index=df.index)
    sweep_low_now = pd.Series(False, index=df.index)
    
    for i in range(1, len(df)):
        if asian_session_ended.iloc[i]:
            swept_high = False
            swept_low = False
        
        if not swept_high and df['high'].iloc[i] > asian_high.iloc[i] and not np.isnan(asian_high.iloc[i]):
            sweep_high_now.iloc[i] = True
            swept_high = True
        
        if not swept_low and df['low'].iloc[i] < asian_low.iloc[i] and not np.isnan(asian_low.iloc[i]):
            sweep_low_now.iloc[i] = True
            swept_low = True
    
    # === Daily Bias Calculation (Candle 2 close vs Candle 1 range) ===
    # Get daily candles for 2 and 3 days back
    df['date'] = df['dt'].apply(lambda x: x.date())
    daily_ohlc = df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    daily_ohlc = daily_ohlc.reset_index()
    
    # Get candle 2 and candle 1 from daily (current day is candle 1)
    c1_high = pd.Series(np.nan, index=df.index)
    c1_low = pd.Series(np.nan, index=df.index)
    c2_close = pd.Series(np.nan, index=df.index)
    c2_high = pd.Series(np.nan, index=df.index)
    c2_low = pd.Series(np.nan, index=df.index)
    
    for idx in range(len(df)):
        current_date = df['date'].iloc[idx]
        if current_date in daily_ohlc['date'].values:
            current_idx_pos = daily_ohlc[daily_ohlc['date'] == current_date].index[0]
            
            # Candle 1 is current daily candle
            # Candle 2 is previous daily candle
            # But Pine uses [1] for c2 and [2] for c1 relative to current daily
            
            # c2Close = close[1], c2High = high[1], c2Low = low[1] (previous day)
            # c1High = high[2], c1Low = low[2] (day before previous)
            
            if current_idx_pos >= 2:
                c2_idx = current_idx_pos - 1
                c1_idx = current_idx_pos - 2
                
                c1_high.iloc[idx] = daily_ohlc['high'].iloc[c1_idx]
                c1_low.iloc[idx] = daily_ohlc['low'].iloc[c1_idx]
                c2_close.iloc[idx] = daily_ohlc['close'].iloc[c2_idx]
                c2_high.iloc[idx] = daily_ohlc['high'].iloc[c2_idx]
                c2_low.iloc[idx] = daily_ohlc['low'].iloc[c2_idx]
    
    # Bullish/Bearish Bias
    bullish_bias = (c2_close > c1_high) | ((c2_low < c1_low) & (c2_close > c1_low))
    bearish_bias = (c2_close < c1_low) | ((c2_high > c1_high) & (c2_close < c1_high))
    
    # === Asia Sweep + Bias Alignment ===
    asia_low_swept_bullish = bullish_bias & sweep_low_now
    asia_high_swept_bearish = bearish_bias & sweep_high_now
    
    # === Kill Zone Windows ===
    london_kz_start_hour = 7
    london_kz_end_hour = 10
    nyam_kz_start_hour = 12
    nyam_kz_end_hour = 15
    
    in_london = (df['hour'] >= london_kz_start_hour) & (df['hour'] < london_kz_end_hour)
    in_nyam = (df['hour'] >= nyam_kz_start_hour) & (df['hour'] < nyam_kz_end_hour)
    in_trading_window = in_london | in_nyam
    
    # === Pattern Definitions ===
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    
    def is_up(offset):
        return close.shift(offset) > open_.shift(offset)
    
    def is_down(offset):
        return close.shift(offset) < open_.shift(offset)
    
    # Stacked OB + FVG
    ob_up = is_down(2) & is_up(1) & (close.shift(1) > high.shift(2))
    ob_down = is_up(2) & is_down(1) & (close.shift(1) < low.shift(2))
    
    fvg_up = low > high.shift(2)
    fvg_down = high < low.shift(2)
    
    stacked_up = ob_up & fvg_up
    stacked_down = ob_down & fvg_down
    
    # === Entry Conditions ===
    long_condition = stacked_up & in_trading_window & asia_low_swept_bullish & (close < asian_high)
    short_condition = stacked_down & in_trading_window & asia_high_swept_bearish & (close > asian_low)
    
    # === Generate Entries ===
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(asian_high.iloc[i]) or pd.isna(asian_low.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': ts_to_dt(df['time'].iloc[i]).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
        
        if short_condition.iloc[i]:
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': ts_to_dt(df['time'].iloc[i]).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
    
    return entries