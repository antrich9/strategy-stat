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
    if len(df) < 3:
        return []

    # Initialize columns
    df = df.copy()
    df['oc2'] = df['open'].shift(2)
    df['cl1'] = df['close'].shift(1)
    df['hi0'] = df['high']
    df['lo0'] = df['low']
    df['lo2'] = df['low'].shift(2)
    df['hi2'] = df['high'].shift(2)
    df['lo1'] = df['low'].shift(1)
    df['hi1'] = df['high'].shift(1)

    # Volume filter
    vol_sma9 = df['volume'].rolling(9).mean()
    df['volfilt'] = vol_sma9 * 1.5

    # ATR filter
    tr = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False).mean()
    df['atr_val'] = atr / 1.5

    # Trend filter
    loc = df['close'].rolling(54).mean()
    loc_prev = loc.shift(1)
    df['loc2'] = loc > loc_prev
    df['locfiltb'] = df['loc2']
    df['locfilts'] = ~df['loc2']

    # FVG conditions
    df['bfvg'] = (df['lo0'] > df['hi2']) & df['volfilt'].fillna(False) & (df['lo0'] - df['hi2'] > df['atr_val']) & df['locfiltb']
    df['sfvg'] = (df['hi0'] < df['lo2']) & df['volfilt'].fillna(False) & (df['lo2'] - df['hi0'] > df['atr_val']) & df['locfilts']

    # Swing detection (simplified - track pivot highs/lows)
    def detect_swings(df, depth=3):
        swing_highs = []
        swing_lows = []
        for i in range(depth, len(df)):
            window_high = df['high'].iloc[i-depth:i+1].max()
            window_low = df['low'].iloc[i-depth:i+1].min()
            if df['high'].iloc[i] == window_high and (i == depth or df['high'].iloc[i-1] != window_high):
                swing_highs.append((df['time'].iloc[i], df['high'].iloc[i]))
            if df['low'].iloc[i] == window_low and (i == depth or df['low'].iloc[i-1] != window_low):
                swing_lows.append((df['time'].iloc[i], df['low'].iloc[i]))
        return swing_highs, swing_lows

    swing_highs, swing_lows = detect_swings(df)

    # Bearish sweeps (high breaks above level, close below)
    bearish_sweep = pd.Series(False, index=df.index)
    for i in range(len(df)):
        for (ts, level) in swing_highs:
            if df['time'].iloc[i] == ts:
                continue
            if df['high'].iloc[i] > level and df['close'].iloc[i] < level:
                bearish_sweep.iloc[i] = True
                break

    # Bullish sweeps (low breaks below level, close above)
    bullish_sweep = pd.Series(False, index=df.index)
    for i in range(len(df)):
        for (ts, level) in swing_lows:
            if df['time'].iloc[i] == ts:
                continue
            if df['low'].iloc[i] < level and df['close'].iloc[i] > level:
                bullish_sweep.iloc[i] = True
                break

    df['bearish_sweep'] = bearish_sweep
    df['bullish_sweep'] = bullish_sweep

    # Entry conditions
    df['long_cond'] = df['bfvg'] | df['bullish_sweep']
    df['short_cond'] = df['sfvg'] | df['bearish_sweep']

    entries = []
    trade_num = 1

    for i in range(len(df)):
        row = df.iloc[i]
        if pd.isna(row['long_cond']) or pd.isna(row['short_cond']):
            continue

        if row['long_cond']:
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
        elif row['short_cond']:
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