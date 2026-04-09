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
    entries = []
    trade_num = 0

    n = len(df)
    if n < 10:
        return entries

    # Extract base series
    time_vals = df['time'].values
    open_vals = df['open'].values
    high_vals = df['high'].values
    low_vals = df['low'].values
    close_vals = df['close'].values
    volume_vals = df['volume'].values

    # Create DataFrames for 4H data
    df_4h = pd.DataFrame({
        'time': df['time'],
        'open': df['open'],
        'high': df['high'],
        'low': df['low'],
        'close': df['close'],
        'volume': df['volume']
    })

    # Resample to 4H
    df_4h['time'] = pd.to_datetime(df_4h['time'], unit='s', utc=True)
    df_4h.set_index('time', inplace=True)
    df_4h_ohlcv = df_4h.resample('240T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    df_4h_ohlcv = df_4h_ohlcv.reset_index()

    # Convert 4H time back to unix timestamp
    df_4h_ohlcv['time'] = df_4h_ohlcv['time'].astype('int64') // 10**9

    # Align 4H data back to 15m bars by forward-filling
    high_4h = pd.Series(index=df.index, dtype=float)
    low_4h = pd.Series(index=df.index, dtype=float)
    close_4h = pd.Series(index=df.index, dtype=float)
    volume_4h = pd.Series(index=df.index, dtype=float)

    for i, row in df_4h_ohlcv.iterrows():
        ts = row['time']
        mask = df['time'] >= ts
        if mask.any():
            idx_vals = df.loc[mask].index
            high_4h.loc[idx_vals] = row['high']
            low_4h.loc[idx_vals] = row['low']
            close_4h.loc[idx_vals] = row['close']
            volume_4h.loc[idx_vals] = row['volume']

    high_4h.fillna(method='ffill', inplace=True)
    low_4h.fillna(method='ffill', inplace=True)
    close_4h.fillna(method='ffill', inplace=True)
    volume_4h.fillna(method='ffill', inplace=True)

    # Daily open - resample to daily
    df_daily = df.copy()
    df_daily['time_dt'] = pd.to_datetime(df_daily['time'], unit='s', utc=True)
    df_daily.set_index('time_dt', inplace=True)
    daily_ohlc = df_daily.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    daily_open = pd.Series(index=df.index, dtype=float)
    for dt, row in daily_ohlc.iterrows():
        ts = int(dt.timestamp())
        mask = df['time'] >= ts
        if mask.any():
            idx_vals = df.loc[mask].index
            daily_open.loc[idx_vals] = row['open']

    daily_open.fillna(method='ffill', inplace=True)

    # Bullish/Bearish allowed
    isDailyGreen = close_vals > daily_open.values
    isDailyRed = close_vals < daily_open.values
    bullishAllowed = isDailyRed
    bearishAllowed = isDailyGreen

    # 4H 240-minute open (session open)
    df_240 = df.copy()
    df_240['time_dt'] = pd.to_datetime(df_240['time'], unit='s', utc=True)
    df_240.set_index('time_dt', inplace=True)
    session_240 = df_240.resample('240T').agg({'open': 'first'}).dropna()
    session_240_open = pd.Series(index=df.index, dtype=float)
    for dt, row in session_240.iterrows():
        ts = int(dt.timestamp())
        mask = df['time'] >= ts
        if mask.any():
            idx_vals = df.loc[mask].index
            session_240_open.loc[idx_vals] = row['open']
    session_240_open.fillna(method='ffill', inplace=True)

    # Detect new 4H candle
    is_new_4h = pd.Series(False, index=df.index)
    prev_session = -1
    for i in range(1, n):
        curr_session = int(df['time'].iloc[i] // (240 * 60))
        if curr_session != prev_session:
            is_new_4h.iloc[i] = True
            prev_session = curr_session

    # Volume filter (4H)
    vol_sma_4h = volume_4h.rolling(9, min_periods=1).mean()
    volfilt = volume_4h.shift(1) > vol_sma_4h.shift(1) * 1.5

    # ATR filter (4H) - Wilder ATR
    tr = pd.concat([
        high_vals - low_vals,
        np.abs(high_vals - np.roll(close_vals, 1)),
        np.abs(low_vals - np.roll(close_vals, 1))
    ], axis=1).max(axis=1)
    tr.iloc[0] = high_vals[0] - low_vals[0]
    atr_4h = pd.Series(0.0, index=df.index)
    atr_period = 20
    alpha = 1.0 / atr_period
    running_atr = np.nan
    for i in range(n):
        if i < 1:
            continue
        tr_val = tr.iloc[i]
        if np.isnan(running_atr):
            running_atr = tr.iloc[max(0, i-atr_period+1):i+1].mean()
        else:
            running_atr = running_atr * (1 - alpha) + tr_val * alpha
        atr_4h.iloc[i] = running_atr
    atr_4h = atr_4h / 1.5
    atrfilt = (low_4h.values - high_4h.shift(2).values > atr_4h.values) | (low_4h.shift(2).values - high_4h.values > atr_4h.values)

    # Trend filter (4H SMA)
    sma_54_4h = close_4h.rolling(54, min_periods=1).mean()
    loc21 = sma_54_4h.values > sma_54_4h.shift(1).values
    locfiltb = loc21
    locfilts = ~loc21

    # Bullish and Bearish FVG using 4H data
    bfvg1 = (low_4h.values > high_4h.shift(2).values) & volfilt.values & atrfilt & locfiltb
    sfvg1 = (high_4h.values < low_4h.shift(2).values) & volfilt.values & atrfilt & locfilts

    # Track last FVG type (1 = Bullish, -1 = Bearish, 0 = None)
    lastFVG = 0

    # Process confirmed bars
    for i in range(2, n):
        if not is_new_4h.iloc[i]:
            continue

        # Detect Sharp Turn in FVGs
        if bfvg1[i] and lastFVG == -1:
            # Bullish Sharp Turn - enter long
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': close_vals[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close_vals[i],
                'raw_price_b': close_vals[i]
            })
            lastFVG = 1
        elif sfvg1[i] and lastFVG == 1:
            # Bearish Sharp Turn - enter short
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': close_vals[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close_vals[i],
                'raw_price_b': close_vals[i]
            })
            lastFVG = -1
        elif bfvg1[i]:
            lastFVG = 1
        elif sfvg1[i]:
            lastFVG = -1

    return entries