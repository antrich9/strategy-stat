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
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)

    # Volume filter: volume[1] > sma(volume, 9) * 1.5
    vol_sma = df['volume'].rolling(9).mean()
    volfilt = df['volume'].shift(1) > vol_sma * 1.5

    # ATR (Wilder's method)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()

    # ATR filter: (low - high[2] > atr) or (low[2] - high > atr)
    atrfilt = ((df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr))

    # Trend filter: SMA(close, 54) > SMA(close, 54)[1]
    sma54 = df['close'].rolling(54).mean()
    loc2 = sma54 > sma54.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    # Bullish FVG: low > high[2]
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb

    # Bearish FVG: high < low[2]
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts

    # Order Block conditions
    # Bullish OB: down[1] and up[0] and close > high[1]
    is_up_0 = df['close'] > df['open']
    is_down_0 = df['close'] < df['open']
    is_up_1 = df['close'].shift(1) > df['open'].shift(1)
    is_down_1 = df['close'].shift(1) < df['open'].shift(1)

    ob_up = is_down_1 & is_up_0 & (df['close'] > df['high'].shift(1))
    ob_down = is_up_1 & is_down_0 & (df['close'] < df['low'].shift(1))

    # Trading windows (London time: 7:45-9:45 and 14:45-16:45)
    minutes = df['time'].dt.hour * 60 + df['time'].dt.minute
    morning_start, morning_end = 465, 585
    afternoon_start, afternoon_end = 885, 1005

    in_trading_window = (
        ((minutes >= morning_start) & (minutes <= morning_end)) |
        ((minutes >= afternoon_start) & (minutes <= afternoon_end))
    )

    # Previous day high/low
    daily_h = df['high'].resample('D').max().ffill()
    daily_l = df['low'].resample('D').min().ffill()
    prev_day_high = daily_h.shift(1).reindex(df.index, method='ffill')
    prev_day_low = daily_l.shift(1).reindex(df.index, method='ffill')

    # Sweep conditions
    # Bullish sweep: price breaks above prev day high then closes back
    bullish_sweep = (df['close'] > prev_day_high) & (df['close'].shift(1) <= prev_day_high)
    # Bearish sweep: price breaks below prev day low then closes back
    bearish_sweep = (df['close'] < prev_day_low) & (df['close'].shift(1) >= prev_day_low)

    # Combined entry conditions
    long_cond = bfvg & ob_up & in_trading_window & bullish_sweep
    short_cond = sfvg & ob_down & in_trading_window & bearish_sweep

    # Generate entries
    entries = []
    trade_num = 0

    for i in range(1, len(df)):
        row = df.iloc[i]

        if long_cond.iloc[i]:
            trade_num += 1
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(row['time'].timestamp() * 1000),
                'entry_time': row['time'].isoformat(),
                'entry_price_guess': float(row['close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(row['close']),
                'raw_price_b': float(row['close'])
            })

        if short_cond.iloc[i]:
            trade_num += 1
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(row['time'].timestamp() * 1000),
                'entry_time': row['time'].isoformat(),
                'entry_price_guess': float(row['close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(row['close']),
                'raw_price_b': float(row['close'])
            })

    return entries