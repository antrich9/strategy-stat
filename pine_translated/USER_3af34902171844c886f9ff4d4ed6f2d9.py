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
    trade_num = 0

    # Daily Open
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    daily_ohlc = df.groupby(pd.Grouper(key='time_dt', freq='D')).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).reset_index()
    daily_open_map = dict(zip(daily_ohlc['time_dt'].dt.date, daily_ohlc['open']))
    df['daily_open'] = df['time_dt'].dt.date.map(daily_open_map)

    # Daily candle color
    df['currentPrice'] = df['close']
    df['isDailyGreen'] = df['currentPrice'] > df['daily_open']
    df['isDailyRed'] = df['currentPrice'] < df['daily_open']

    # Previous Day High/Low
    df['date'] = df['time_dt'].dt.date
    daily_hilo = df.groupby('date').agg({'high': 'max', 'low': 'min'}).reset_index()
    daily_hilo.columns = ['date', 'dh', 'dl']
    daily_hilo['prev_pdh'] = daily_hilo['dh'].shift(1)
    daily_hilo['prev_pdl'] = daily_hilo['dl'].shift(1)
    prev_day_map_pdh = dict(zip(daily_hilo['date'], daily_hilo['prev_pdh']))
    prev_day_map_pdl = dict(zip(daily_hilo['date'], daily_hilo['prev_pdl']))
    df['pdh'] = df['date'].map(prev_day_map_pdh)
    df['pdl'] = df['date'].map(prev_day_map_pdl)

    # New day detection
    df['newDay'] = df['daily_open'].diff().fillna(0) != 0

    # Bias calculation
    df['sweptLow'] = df['low'] < df['pdl']
    df['sweptHigh'] = df['high'] > df['pdh']
    df['brokeHigh'] = df['close'] > df['pdh']
    df['brokeLow'] = df['close'] < df['pdl']

    bias = 0
    df['bias'] = 0.0
    for i in range(len(df)):
        if pd.isna(df['pdh'].iloc[i]) or pd.isna(df['pdl'].iloc[i]):
            df.loc[df.index[i], 'bias'] = 0
            bias = 0
            continue
        if df['newDay'].iloc[i]:
            bias = 0
        if df['sweptLow'].iloc[i] and df['brokeHigh'].iloc[i]:
            bias = 1
        elif df['sweptHigh'].iloc[i] and df['brokeLow'].iloc[i]:
            bias = -1
        elif df['low'].iloc[i] < df['pdl'].iloc[i]:
            bias = -1
        elif df['high'].iloc[i] > df['pdh'].iloc[i]:
            bias = 1
        df.loc[df.index[i], 'bias'] = bias

    # 4H data via resampling
    df_4h = df.set_index('time_dt').resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).reset_index()
    df_4h['high_4h1'] = df_4h['high']
    df_4h['low_4h1'] = df_4h['low']
    df_4h['close_4h1'] = df_4h['close']
    df_4h['volume_4h1'] = df_4h['volume']

    # 4H ATR
    tr_4h = df_4h['high'].sub(df_4h['low']).join(df_4h['high'].sub(df_4h['close'].shift(1)).abs().clip(lower=0), how='outer')
    tr_4h = tr_4h.join(df_4h['low'].sub(df_4h['close'].shift(1)).abs().clip(lower=0), how='outer').max(axis=1)
    atr_4h_series = tr_4h.ewm(alpha=1/14, adjust=False).mean()
    df_4h['atr_4h1'] = atr_4h_series / 1.5

    # 4H SMA for volume
    vol_4h_sma = df_4h['volume_4h1'].rolling(9).mean() * 1.5

    # 4H SMA for close (trend)
    loc1 = df_4h['close_4h1'].rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)

    # Bullish FVG
    bfvg1 = (df_4h['low_4h1'] > df_4h['high_4h1'].shift(2)) & (df_4h['volume_4h1'].shift(1) > vol_4h_sma.shift(1)) & ((df_4h['low_4h1'].shift(1) - df_4h['high_4h1'].shift(2) > df_4h['atr_4h1'].shift(1)) | (df_4h['low_4h1'].shift(2) - df_4h['high_4h1'].shift(1) > df_4h['atr_4h1'].shift(1))) & loc21

    # Bearish FVG
    sfvg1 = (df_4h['high_4h1'] < df_4h['low_4h1'].shift(2)) & (df_4h['volume_4h1'].shift(1) > vol_4h_sma.shift(1)) & ((df_4h['low_4h1'].shift(1) - df_4h['high_4h1'].shift(2) > df_4h['atr_4h1'].shift(1)) | (df_4h['low_4h1'].shift(2) - df_4h['high_4h1'].shift(1) > df_4h['atr_4h1'].shift(1))) & ~loc21

    # Map 4H FVG to 15m bars
    df['bfvg_4h'] = df['time_dt'].map(dict(zip(df_4h['time_dt'], bfvg1))) | df['time_dt'].map(dict(zip(df_4h['time_dt'].shift(1), bfvg1.shift(1))))
    df['sfvg_4h'] = df['time_dt'].map(dict(zip(df_4h['time_dt'], sfvg1))) | df['time_dt'].map(dict(zip(df_4h['time_dt'].shift(1), sfvg1.shift(1))))

    # New 4H candle detection
    df['is_new_4h1'] = df['time_dt'].dt.hour.isin([0, 4, 8, 12, 16, 20]) & (df['time_dt'].dt.minute == 0)
    df['is_new_4h1'] = df['is_new_4h1'] | (df['time_dt'].dt.hour == 0) | ((df['time_dt'].dt.hour % 4 == 0) & (df['time_dt'].dt.hour > 0) & (df['time_dt'].dt.minute == 0))

    # Last FVG type tracking
    lastFVG = 0
    # Entry conditions
    for i in range(1, len(df)):
        if pd.isna(df['pdh'].iloc[i]) or pd.isna(df['pdl'].iloc[i]):
            continue
        curr_bfvg = df['bfvg_4h'].iloc[i] if i < len(df_4h) * 4 else False
        curr_sfvg = df['sfvg_4h'].iloc[i] if i < len(df_4h) * 4 else False
        is_new_4h = df['is_new_4h1'].iloc[i] if 'is_new_4h1' in df.columns else False
        if not is_new_4h:
            continue
        bias_val = df['bias'].iloc[i]
        if curr_bfvg and lastFVG == -1 and bias_val == 1:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            lastFVG = 1
        elif curr_sfvg and lastFVG == 1 and bias_val == -1:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            lastFVG = -1
        elif curr_bfvg:
            lastFVG = 1
        elif curr_sfvg:
            lastFVG = -1

    return results