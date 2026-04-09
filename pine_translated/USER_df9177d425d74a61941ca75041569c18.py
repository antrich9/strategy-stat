import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    entries = []
    trade_num = 1

    # Get daily open for comparison (resample to daily)
    df['date'] = pd.to_datetime(df['time'], unit='s').dt.date
    daily_ohlc = df.groupby('date').agg({'open': 'first'}).reset_index()
    daily_ohlc.columns = ['date', 'daily_open']

    df = df.merge(daily_ohlc, on='date', how='left')

    # Current price vs daily open - live color of daily candle
    isDailyGreen = df['close'] > df['daily_open']
    isDailyRed = df['close'] < df['daily_open']

    # Trade filters
    bullishAllowed = isDailyRed
    bearishAllowed = isDailyGreen

    # Resample to 4H for FVG detection
    df['hour'] = pd.to_datetime(df['time'], unit='s').dt.hour
    df['4h_date'] = df['date'].astype(str) + '_' + ((df['hour'] // 4) * 4).astype(str)

    h4_ohlc = df.groupby('4h_date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'open': 'first',
        'volume': 'sum',
        'time': 'last'
    }).reset_index()

    # 4H ATR (Wilder)
    high_4h = h4_ohlc['high']
    low_4h = h4_ohlc['low']
    close_4h = h4_ohlc['close'].shift(1)

    tr1 = high_4h - low_4h
    tr2 = (high_4h - close_4h).abs()
    tr3 = (low_4h - close_4h).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr.rolling(20).mean()

    # 4H SMA for trend (54)
    sma54 = close_4h.rolling(54).mean()
    sma54_prev = sma54.shift(1)
    loc21 = sma54 > sma54_prev

    # Bullish FVG: low > high[2]
    bfvg1 = (low_4h > high_4h.shift(2))

    # Bearish FVG: high < low[2]
    sfvg1 = (high_4h < low_4h.shift(2))

    # Apply filters
    volfilt1 = True  # inp11 = false
    atrfilt1 = True  # inp21 = false
    locfiltb1 = True  # inp31 = false
    locfilts1 = True  # inp31 = false

    bfvg = bfvg1 & volfilt1 & atrfilt1 & locfiltb1
    sfvg = sfvg1 & volfilt1 & atrfilt1 & locfilts1

    # Create 4H signal Series
    h4_signals = pd.DataFrame({
        '4h_date': h4_ohlc['4h_date'],
        'bfvg': bfvg.values,
        'sfvg': sfvg.values
    })

    df = df.merge(h4_signals, on='4h_date', how='left')
    df['bfvg'] = df['bfvg'].fillna(False)
    df['sfvg'] = df['sfvg'].fillna(False)

    # Entry conditions
    # Long: bullishAllowed AND bfvg
    # Short: bearishAllowed AND sfvg
    long_cond = bullishAllowed & df['bfvg']
    short_cond = bearishAllowed & df['sfvg']

    # Iterate and generate entries
    for i in range(len(df)):
        if long_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries