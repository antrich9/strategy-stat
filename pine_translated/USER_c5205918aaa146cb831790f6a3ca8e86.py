import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

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
    if len(df) < 20:
        return []

    # Helper functions
    def is_bst(dt):
        year = dt.year
        # Last Sunday of March at 1:00 AM UTC
        mar31 = datetime(year, 3, 31)
        last_sun_mar = mar31 - timedelta(days=(mar31.weekday() + 1) % 7)
        # Last Sunday of October at 1:00 AM UTC
        oct31 = datetime(year, 10, 31)
        last_sun_oct = oct31 - timedelta(days=(oct31.weekday() + 1) % 7)
        bst_start = datetime(year, last_sun_mar.year, last_sun_mar.month, last_sun_mar.day, 1, 0)
        bst_end = datetime(year, last_sun_oct.year, last_sun_oct.month, last_sun_oct.day, 1, 0)
        return bst_start <= dt < bst_end

    def to_london_time(ts):
        utc_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        offset_hours = 1 if is_bst(utc_dt) else 0
        london_dt = utc_dt + timedelta(hours=offset_hours)
        return utc_dt, london_dt

    def resample_to_4h(data):
        df_copy = data.copy()
        df_copy['ts'] = pd.to_datetime(df_copy['time'], unit='s')
        df_copy = df_copy.set_index('ts')
        resampled = df_copy.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        return resampled.reset_index()

    def wilder_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def wilder_atr(high, low, close, period=20):
        tr = np.maximum(high - low, np.maximum(
            np.abs(high - close.shift(1)),
            np.abs(low - close.shift(1))
        ))
        atr = tr.ewm(alpha=1.0/period, adjust=False).mean()
        return atr

    # Resample to 4H
    df_4h = resample_to_4h(df)
    if len(df_4h) < 5:
        return []

    # Calculate daily swing highs/lows using daily data from 4H
    # Swing high: dailyHigh21 < dailyHigh22 and dailyHigh11[3] < dailyHigh22 and dailyHigh11[4] < dailyHigh22
    # Swing low: dailyLow21 > dailyLow22 and dailyLow11[3] > dailyLow22 and dailyLow11[4] > dailyLow22
    
    # For daily highs/lows, we need to aggregate 4H to daily
    df_4h_copy = df_4h.copy()
    df_4h_copy['date'] = df_4h_copy['ts'].dt.date
    daily = df_4h_copy.groupby('date').agg({
        'high': 'max',
        'low': 'min'
    })
    daily['dailyHigh11'] = daily['high']
    daily['dailyLow11'] = daily['low']
    daily['dailyHigh21'] = daily['high'].shift(1)
    daily['dailyLow21'] = daily['low'].shift(1)
    daily['dailyHigh22'] = daily['high'].shift(2)
    daily['dailyLow22'] = daily['low'].shift(2)
    daily = daily.dropna()

    is_swing_high = (daily['dailyHigh21'] < daily['dailyHigh22']) & \
                    (daily['dailyHigh11'].shift(3) < daily['dailyHigh22']) & \
                    (daily['dailyHigh11'].shift(4) < daily['dailyHigh22'])
    is_swing_low = (daily['dailyLow21'] > daily['dailyLow22']) & \
                   (daily['dailyLow11'].shift(3) > daily['dailyLow22']) & \
                   (daily['dailyLow11'].shift(4) > daily['dailyLow22'])

    last_swing_type = pd.Series('none', index=daily.index)
    last_swing_high = pd.Series(np.nan, index=daily.index, dtype=float)
    last_swing_low = pd.Series(np.nan, index=daily.index, dtype=float)

    for i in range(len(daily)):
        if is_swing_high.iloc[i]:
            last_swing_high.iloc[i] = daily['dailyHigh22'].iloc[i]
            last_swing_type.iloc[i] = 'dailyHigh'
        elif is_swing_low.iloc[i]:
            last_swing_low.iloc[i] = daily['dailyLow22'].iloc[i]
            last_swing_type.iloc[i] = 'dailyLow'
        elif i > 0:
            last_swing_high.iloc[i] = last_swing_high.iloc[i-1]
            last_swing_low.iloc[i] = last_swing_low.iloc[i-1]
            last_swing_type.iloc[i] = last_swing_type.iloc[i-1]

    # Merge daily swing info back to 4H
    df_4h_copy = df_4h_copy.set_index('date')
    df_4h_copy['lastSwingType11'] = last_swing_type
    df_4h_copy['last_swing_high_4h'] = last_swing_high
    df_4h_copy['last_swing_low_4h'] = last_swing_low
    df_4h = df_4h_copy.reset_index(drop=True)

    # Calculate 4H indicators
    df_4h['high_4h'] = df_4h['high']
    df_4h['low_4h'] = df_4h['low']
    df_4h['close_4h'] = df_4h['close']
    df_4h['high_4h_1'] = df_4h['high'].shift(1)
    df_4h['low_4h_1'] = df_4h['low'].shift(1)
    df_4h['high_4h_2'] = df_4h['high'].shift(2)
    df_4h['low_4h_2'] = df_4h['low'].shift(2)

    # FVG conditions
    df_4h['bfvg_condition'] = df_4h['low_4h'] > df_4h['high_4h_2']
    df_4h['sfvg_condition'] = df_4h['high_4h'] < df_4h['low_4h_2']

    # RSI on 4H close
    df_4h['rsi_4h'] = wilder_rsi(df_4h['close_4h'], 14)
    df_4h['atr_4h'] = wilder_atr(df_4h['high'], df_4h['low'], df_4h['close'], 20) / 1.5

    # Trading windows (London time)
    london_window1_start = pd.to_datetime('07:45:00').time()
    london_window1_end = pd.to_datetime('11:45:00').time()
    london_window2_start = pd.to_datetime('14:00:00').time()
    london_window2_end = pd.to_datetime('14:45:00').time()

    df['utc_dt'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC')
    df['london_dt'] = df['utc_dt'].dt.tz_convert('Europe/London')
    df['time_only'] = df['london_dt'].dt.time

    in_window1 = (df['time_only'] >= london_window1_start) & (df['time_only'] < london_window1_end)
    in_window2 = (df['time_only'] >= london_window2_start) & (df['time_only'] < london_window2_end)
    df['in_trading_window'] = in_window1 | in_window2

    # Map 4H conditions back to 15min bars
    df_4h['ts_floor'] = df_4h['ts'].dt.floor('15min')
    df_4h_subset = df_4h[['ts_floor', 'bfvg_condition', 'sfvg_condition', 'lastSwingType11', 
                         'high_4h', 'low_4h', 'high_4h_1', 'low_4h_1', 'high_4h_2', 'low_4h_2',
                         'rsi_4h', 'atr_4h']].copy()
    df_merged = df.merge(df_4h_subset, left_on='time', right_on=df_4h_subset['ts_floor'].astype(int) // 10**9, how='left')
    df_merged['bfvg_condition'] = df_merged['bfvg_condition'].fillna(False)
    df_merged['sfvg_condition'] = df_merged['sfvg_condition'].fillna(False)
    df_merged['lastSwingType11'] = df_merged['lastSwingType11'].fillna('none')
    df_merged['in_trading_window'] = df_merged['in_trading_window'].fillna(False)

    # Entry conditions
    bull_entry = df_merged['bfvg_condition'] & (df_merged['lastSwingType11'] == 'dailyLow')
    bear_entry = df_merged['sfvg_condition'] & (df_merged['lastSwingType11'] == 'dailyHigh')

    entries = []
    trade_num = 1

    for i in range(1, len(df_merged)):
        row = df_merged.iloc[i]
        prev_row = df_merged.iloc[i-1]

        if bull_entry.iloc[i] and not bull_entry.iloc[i-1]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(row['time']),
                'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(row['close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(row['close']),
                'raw_price_b': float(row['close'])
            })
            trade_num += 1

        if bear_entry.iloc[i] and not bear_entry.iloc[i-1]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(row['time']),
                'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(row['close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(row['close']),
                'raw_price_b': float(row['close'])
            })
            trade_num += 1

    return entries