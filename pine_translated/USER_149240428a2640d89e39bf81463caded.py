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
    
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # EMA lengths from Pine Script
    ema1_len = 8
    ema2_len = 20
    ema3_len = 50
    
    # Calculate EMAs on 15m data
    ema1_15 = data['close'].ewm(span=ema1_len, adjust=False).mean()
    ema2_15 = data['close'].ewm(span=ema2_len, adjust=False).mean()
    ema3_15 = data['close'].ewm(span=ema3_len, adjust=False).mean()
    
    # Bullish EMA condition on 15m: ema1 > ema2 > ema3
    bullish_ema_15 = (ema1_15 > ema2_15) & (ema2_15 > ema3_15)
    # Bearish EMA condition on 15m: ema1 < ema2 < ema3
    bearish_ema_15 = (ema1_15 < ema2_15) & (ema2_15 < ema3_15)
    
    # For W/D/4H/1H EMAs, we need to resample to higher timeframes
    # Then calculate EMAs on those timeframes
    
    # Daily resampling
    data['date'] = pd.to_datetime(data['time'], unit='s', utc=True).dt.date
    daily_ohlc = data.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'time': 'first'
    })
    daily_ohlc['ema1'] = daily_ohlc['close'].ewm(span=ema1_len, adjust=False).mean()
    daily_ohlc['ema2'] = daily_ohlc['close'].ewm(span=ema2_len, adjust=False).mean()
    daily_ohlc['ema3'] = daily_ohlc['close'].ewm(span=ema3_len, adjust=False).mean()
    daily_ohlc['bullish'] = (daily_ohlc['ema1'] > daily_ohlc['ema2']) & (daily_ohlc['ema2'] > daily_ohlc['ema3'])
    daily_ohlc['bearish'] = (daily_ohlc['ema1'] < daily_ohlc['ema2']) & (daily_ohlc['ema2'] < daily_ohlc['ema3'])
    
    # 4H resampling (240 min)
    data['time_dt'] = pd.to_datetime(data['time'], unit='s', utc=True)
    data['4h_bucket'] = data['time_dt'].dt.floor('4H')
    four_h_ohlc = data.groupby('4h_bucket').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'time': 'first'
    })
    four_h_ohlc['ema1'] = four_h_ohlc['close'].ewm(span=ema1_len, adjust=False).mean()
    four_h_ohlc['ema2'] = four_h_ohlc['close'].ewm(span=ema2_len, adjust=False).mean()
    four_h_ohlc['ema3'] = four_h_ohlc['close'].ewm(span=ema3_len, adjust=False).mean()
    four_h_ohlc['bullish'] = (four_h_ohlc['ema1'] > four_h_ohlc['ema2']) & (four_h_ohlc['ema2'] > four_h_ohlc['ema3'])
    four_h_ohlc['bearish'] = (four_h_ohlc['ema1'] < four_h_ohlc['ema2']) & (four_h_ohlc['ema2'] < four_h_ohlc['ema3'])
    
    # 1H resampling
    data['1h_bucket'] = data['time_dt'].dt.floor('1H')
    one_h_ohlc = data.groupby('1h_bucket').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'time': 'first'
    })
    one_h_ohlc['ema1'] = one_h_ohlc['close'].ewm(span=ema1_len, adjust=False).mean()
    one_h_ohlc['ema2'] = one_h_ohlc['close'].ewm(span=ema2_len, adjust=False).mean()
    one_h_ohlc['ema3'] = one_h_ohlc['close'].ewm(span=ema3_len, adjust=False).mean()
    one_h_ohlc['bullish'] = (one_h_ohlc['ema1'] > one_h_ohlc['ema2']) & (one_h_ohlc['ema2'] > one_h_ohlc['ema3'])
    one_h_ohlc['bearish'] = (one_h_ohlc['ema1'] < one_h_ohlc['ema2']) & (one_h_ohlc['ema2'] < one_h_ohlc['ema3'])
    
    # Weekly resampling
    data['week'] = data['time_dt'].dt.isocalendar().week
    data['year'] = data['time_dt'].dt.year
    data['week_bucket'] = data['year'].astype(str) + '-W' + data['week'].astype(str)
    weekly_ohlc = data.groupby('week_bucket').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'time': 'first'
    })
    weekly_ohlc['ema1'] = weekly_ohlc['close'].ewm(span=ema1_len, adjust=False).mean()
    weekly_ohlc['ema2'] = weekly_ohlc['close'].ewm(span=ema2_len, adjust=False).mean()
    weekly_ohlc['ema3'] = weekly_ohlc['close'].ewm(span=ema3_len, adjust=False).mean()
    weekly_ohlc['bullish'] = (weekly_ohlc['ema1'] > weekly_ohlc['ema2']) & (weekly_ohlc['ema2'] > weekly_ohlc['ema3'])
    weekly_ohlc['bearish'] = (weekly_ohlc['ema1'] < weekly_ohlc['ema2']) & (weekly_ohlc['ema2'] < weekly_ohlc['ema3'])
    
    # Create boolean series for each timeframe EMA condition aligned back to 15m data
    data['date'] = pd.to_datetime(data['time'], unit='s', utc=True).dt.date
    data = data.merge(daily_ohlc[['bullish', 'bearish']].rename(columns={'bullish': 'bullish_d', 'bearish': 'bearish_d'}), 
                      left_on='date', right_index=True, how='left')
    
    data['4h_bucket_dt'] = pd.to_datetime(data['time'], unit='s', utc=True).dt.floor('4H')
    data = data.merge(four_h_ohlc[['bullish', 'bearish']].rename(columns={'bullish': 'bullish_4h', 'bearish': 'bearish_4h'}), 
                      left_on='4h_bucket_dt', right_index=True, how='left')
    
    data['1h_bucket_dt'] = pd.to_datetime(data['time'], unit='s', utc=True).dt.floor('1H')
    data = data.merge(one_h_ohlc[['bullish', 'bearish']].rename(columns={'bullish': 'bullish_1h', 'bearish': 'bearish_1h'}), 
                      left_on='1h_bucket_dt', right_index=True, how='left')
    
    # Forward fill weekly to 15m
    weekly_ohlc['week_start'] = pd.to_datetime(weekly_ohlc['time'], unit='s', utc=True).dt.floor('1W')
    weekly_merged = weekly_ohlc[['bullish', 'bearish', 'week_start']].copy()
    weekly_merged = weekly_merged.rename(columns={'bullish': 'bullish_w', 'bearish': 'bearish_w'})
    
    data['week_dt'] = pd.to_datetime(data['time'], unit='s', utc=True).dt.floor('1W')
    data = data.merge(weekly_merged[['bullish_w', 'bearish_w']], left_on='week_dt', right_on=weekly_merged['week_start'], how='left')
    
    # Fill NaN with previous value for weekly
    data['bullish_w'] = data['bullish_w'].ffill()
    data['bearish_w'] = data['bearish_w'].ffill()
    data['bullish_d'] = data['bullish_d'].ffill()
    data['bearish_d'] = data['bearish_d'].ffill()
    data['bullish_4h'] = data['bullish_4h'].ffill()
    data['bearish_4h'] = data['bearish_4h'].ffill()
    data['bullish_1h'] = data['bullish_1h'].ffill()
    data['bearish_1h'] = data['bearish_1h'].ffill()
    
    # Combined EMA conditions
    bullish_ema_all = (data['bullish_w'] & data['bullish_d'] & data['bullish_4h'] & data['bullish_1h'] & bullish_ema_15.values)
    bearish_ema_all = (data['bearish_w'] & data['bearish_d'] & data['bearish_4h'] & data['bearish_1h'] & bearish_ema_15.values)
    
    # Detect new day
    data['prev_date'] = data['date'].shift(1)
    data['new_day'] = data['date'] != data['prev_date']
    data.loc[data['prev_date'].isna(), 'new_day'] = False
    
    # Calculate previous day high/low manually
    pd_high = np.nan
    pd_low = np.nan
    temp_high = np.nan
    temp_low = np.nan
    swept_high = False
    swept_low = False
    
    prev_day_high = np.full(len(data), np.nan)
    prev_day_low = np.full(len(data), np.nan)
    swept_high_arr = np.full(len(data), False)
    swept_low_arr = np.full(len(data), False)
    sweep_high_triggered = np.full(len(data), False)
    sweep_low_triggered = np.full(len(data), False)
    
    for i in range(len(data)):
        if data['new_day'].iloc[i]:
            pd_high = temp_high if not np.isnan(temp_high) else pd_high
            pd_low = temp_low if not np.isnan(temp_low) else pd_low
            temp_high = data['high'].iloc[i]
            temp_low = data['low'].iloc[i]
            swept_high = False
            swept_low = False
        else:
            temp_high = data['high'].iloc[i] if np.isnan(temp_high) else max(temp_high, data['high'].iloc[i])
            temp_low = data['low'].iloc[i] if np.isnan(temp_low) else min(temp_low, data['low'].iloc[i])
        
        prev_day_high[i] = pd_high
        prev_day_low[i] = pd_low
        swept_high_arr[i] = swept_high
        swept_low_arr[i] = swept_low
        
        # Sweep detection
        if not swept_high and data['high'].iloc[i] > pd_high:
            sweep_high_triggered[i] = True
            swept_high = True
        if not swept_low and data['low'].iloc[i] < pd_low:
            sweep_low_triggered[i] = True
            swept_low = True
    
    data['prev_day_high'] = prev_day_high
    data['prev_day_low'] = prev_day_low
    data['swept_high'] = swept_high_arr
    data['swept_low'] = swept_low_arr
    data['sweep_high_triggered'] = sweep_high_triggered
    data['sweep_low_triggered'] = sweep_low_triggered
    
    # London trading windows (simplified - assumes data is in UTC)
    # Window 1: 07:45 - 11:45 London time
    # Window 2: 14:00 - 14:45 London time
    data['hour'] = pd.to_datetime(data['time'], unit='s', utc=True).dt.hour
    data['minute'] = pd.to_datetime(data['time'], unit='s', utc=True).dt.minute
    data['time_minutes'] = data['hour'] * 60 + data['minute']
    
    # London time is UTC in winter, UTC+1 in summer
    # Approximate: May-Sep is summer time (UTC+1)
    data['month'] = pd.to_datetime(data['time'], unit='s', utc=True).dt.month
    data['london_offset'] = 0  # Simplified - using UTC directly
    data['london_time'] = (data['time_minutes'] - data['london_offset']) % 1440
    data['in_window1'] = (data['london_time'] >= 7*60 + 45) & (data['london_time'] < 11*60 + 45)
    data['in_window2'] = (data['london_time'] >= 14*60) & (data['london_time'] < 14*60 + 45)
    data['in_trading_window'] = data['in_window1'] | data['in_window2']
    
    # 4H FVG calculation
    # FVG: gap between current 4H low and 4H high 2 bars ago
    data['4h_high_shifted'] = data['high'].shift(1).rolling(240).max().shift(-1)  # Approximate 4H high
    data['4h_low_shifted'] = data['low'].shift(1).rolling(240).min().shift(-1)   # Approximate 4H low
    
    # More accurate: use 4H resampled data for FVG
    four_h_ohlc['high_1'] = four_h_ohlc['high'].shift(1)
    four_h_ohlc['low_1'] = four_h_ohlc['low'].shift(1)
    four_h_ohlc['high_2'] = four_h_ohlc['high'].shift(2)
    four_h_ohlc['low_2'] = four_h_ohlc['low'].shift(2)
    
    # Bullish FVG: current 4H low > 4H high 2 bars ago
    bfvg_4h = four_h_ohlc['low'] > four_h_ohlc['high_2']
    # Bearish FVG: current 4H high < 4H low 2 bars ago
    sfvg_4h = four_h_ohlc['high'] < four_h_ohlc['low_2']
    
    # Map 4H FVG back to 15m
    data['4h_bucket_dt'] = pd.to_datetime(data['time'], unit='s', utc=True).dt.floor('4H')
    bfvg_4h_aligned = four_h_ohlc['low'] > four_h_ohlc['high_2']
    sfvg_4h_aligned = four_h_ohlc['high'] < four_h_ohlc['low_2']
    
    data = data.merge(
        pd.DataFrame({
            '4h_bucket_dt': four_h_ohlc.index,
            'bfvg_4h': bfvg_4h_aligned.values,
            'sfvg_4h': sfvg_4h_aligned.values
        }),
        on='4h_bucket_dt',
        how='left'
    )
    data['bfvg_4h'] = data['bfvg_4h'].ffill()
    data['sfvg_4h'] = data['sfvg_4h'].ffill()
    
    # 4H Volume filter (sma of volume on 4H)
    four_h_ohlc['volume_sma_9'] = four_h_ohlc['volume'].rolling(9).mean()
    four_h_ohlc['volume_filter'] = four_h_ohlc['volume'] > four_h_ohlc['volume_sma_9'] * 1.5
    
    data = data.merge(
        pd.DataFrame({
            '4h_bucket_dt': four_h_ohlc.index,
            'volume_filter_4h': four_h_ohlc['volume_filter'].values
        }),
        on='4h_bucket_dt',
        how='left'
    )
    data['volume_filter_4h'] = data['volume_filter_4h'].ffill().fillna(False)
    
    # ATR filter on 4H
    # Calculate ATR manually using Wilder's method
    high_4h = four_h_ohlc['high']
    low_4h = four_h_ohlc['low']
    close_4h_prev = four_h_ohlc['close'].shift(1)
    
    tr_4h = np.maximum(
        high_4h - low_4h,
        np.maximum(
            np.abs(high_4h - close_4h_prev),
            np.abs(low_4h - close_4h_prev)
        )
    )
    
    # Wilder ATR
    atr_4h = np.zeros(len(tr_4h))
    atr_4h[0] = tr_4h.iloc[0]
    for i in range(1, len(tr_4h)):
        atr_4h[i] = (atr_4h[i-1] * 19 + tr_4h.iloc[i]) / 20
    
    four_h_ohlc['atr_20'] = atr_4h
    four_h_ohlc['atr_threshold'] = four_h_ohlc['atr_20'] / 1.5
    
    # ATR filter: (low_4h - high_4h_2 > atr_4h) or (low_4h_2 - high_4h > atr_4h)
    atr_filt_bull = (four_h_ohlc['low'] - four_h_ohlc['high'].shift(2) > four_h_ohlc['atr_threshold'])
    atr_filt_bear = (four_h_ohlc['low'].shift(2) - four_h_ohlc['high'] > four_h_ohlc['atr_threshold'])
    atr_filter = atr_filt_bull | atr_filt_bear
    
    data = data.merge(
        pd.DataFrame({
            '4h_bucket_dt': four_h_ohlc.index,
            'atr_filter': atr_filter.values
        }),
        on='4h_bucket_dt',
        how='left'
    )
    data['atr_filter'] = data['atr_filter'].ffill().fillna(False)
    
    # Trend filter using 4H SMA(close, 54)
    four_h_ohlc['sma_54'] = four_h_ohlc['close'].rolling(54).mean()
    four_h_ohlc['trend_up'] = four_h_ohlc['sma_54'] > four_h_ohlc['sma_54'].shift(1)
    
    data = data.merge(
        pd.DataFrame({
            '4h_bucket_dt': four_h_ohlc.index,
            'trend_up_4h': four_h_ohlc['trend_up'].values
        }),
        on='4h_bucket_dt',
        how='left'
    )
    data['trend_up_4h'] = data['trend_up_4h'].ffill().fillna(False)
    
    # Entry conditions (based on filters being optional in Pine Script, using defaults)
    # inp11 = false (no volume filter), inp21 = false (no ATR filter), inp31 = false (no trend filter)
    vol_filter = data['volume_filter_4h']  # Will be ignored if inp11 = false
    atr_filter_cond = data['atr_filter']   # Will be ignored if inp21 = false
    trend_filter_bull = data['trend_up_4h']  # Will be ignored if inp31 = false
    trend_filter_bear = ~data['trend_up_4h']  # Will be ignored if inp31 = false
    
    # Combine all entry conditions for LONG
    long_entry = (
        bullish_ema_all & 
        data['bfvg_4h'] & 
        ~data['swept_high'] &
        data['in_trading_window']
    )
    
    # Combine all entry conditions for SHORT
    short_entry = (
        bearish_ema_all & 
        data['sfvg_4h'] & 
        ~data['swept_low'] &
        data['in_trading_window']
    )
    
    # Skip bars where EMAs are NaN
    valid_ema = ~(ema1_15.isna() | ema2_15.isna() | ema3_15.isna())
    long_entry = long_entry & valid_ema
    short_entry = short_entry & valid_ema
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(data)):
        entry_price = data['close'].iloc[i]
        ts = int(data['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        if long_entry.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
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
        
        if short_entry.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
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