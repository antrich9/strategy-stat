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
    trade_num = 1

    # Convert time to datetime for easier manipulation
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)

    # Create 4H resampled data
    df['4h_period'] = (df['datetime'].dt.hour // 4) * 4 + df['datetime'].dt.day * 24 + (df['datetime'].dt.minute // 60) * 24

    # Group by 4H period and take first and last values
    def first(x):
        return x.iloc[0]
    def last(x):
        return x.iloc[-1]

    agg_dict = {
        'time': 'first',
        'datetime': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    # Create 4H OHLCV
    df_4h = df.groupby('4h_period').agg(agg_dict).reset_index(drop=True)
    df_4h = df_4h.sort_values('datetime').reset_index(drop=True)

    if len(df_4h) < 3:
        return entries

    # Calculate 4H indicators
    # Volume Filter - 4H data
    volfilt_4h = df_4h['volume'].shift(1) > df_4h['volume'].shift(1).rolling(9).mean() * 1.5

    # ATR Filter - Wilder ATR on 4H
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    close_4h = df_4h['close']
    prev_close_4h = close_4h.shift(1)

    # True Range
    tr1 = high_4h - low_4h
    tr2 = abs(high_4h - prev_close_4h)
    tr3 = abs(low_4h - prev_close_4h)
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder ATR
    atr_length = 20
    atr_4h_raw = tr_4h.ewm(alpha=1.0/atr_length, adjust=False).mean()
    atr_4h = atr_4h_raw / 1.5

    atrfilt_4h = (low_4h - high_4h.shift(2) > atr_4h) | (low_4h.shift(2) - high_4h > atr_4h)

    # Trend Filter - SMA on 4H close
    loc1_4h = close_4h.rolling(54).mean()
    loc2_4h = loc1_4h > loc1_4h.shift(1)
    locfiltb_4h = loc2_4h
    locfilts_4h = ~loc2_4h

    # Identify Bullish and Bearish FVGs - using 4H data
    bfvg_4h = (low_4h > high_4h.shift(2)) & volfilt_4h & atrfilt_4h & locfiltb_4h
    sfvg_4h = (high_4h < low_4h.shift(2)) & volfilt_4h & atrfilt_4h & locfilts_4h

    # Track last FVG type
    lastFVG = 0

    # Detect 4H candle changes
    df_4h['prev_period'] = df_4h['4h_period'].shift(1)
    is_new_4h_candle = df_4h['4h_period'] != df_4h['prev_period']

    # Build signal series aligned with original df
    df['sharp_turn_long'] = False
    df['sharp_turn_short'] = False

    # Create mapping from 4H datetime to original df index range
    df['4h_idx'] = df_4h.index

    # Get 4H candle signals
    for i in range(2, len(df_4h)):
        if is_new_4h_candle.iloc[i] or i == 2:
            if bfvg_4h.iloc[i] and lastFVG == -1:
                # Bullish Sharp Turn - LONG entry
                signal_time = df_4h['time'].iloc[i]
                # Find the first bar in original df that corresponds to this 4H candle
                mask = (df['time'] >= signal_time) & (df['time'] < signal_time + 14400)
                entry_indices = df[mask].index.tolist()
                if entry_indices:
                    entry_idx = entry_indices[0]
                    entry_ts = int(df['time'].iloc[entry_idx])
                    entry_price = df['close'].iloc[entry_idx]
                    entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                    entries.append({
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
                lastFVG = 1
            elif sfvg_4h.iloc[i] and lastFVG == 1:
                # Bearish Sharp Turn - SHORT entry
                signal_time = df_4h['time'].iloc[i]
                mask = (df['time'] >= signal_time) & (df['time'] < signal_time + 14400)
                entry_indices = df[mask].index.tolist()
                if entry_indices:
                    entry_idx = entry_indices[0]
                    entry_ts = int(df['time'].iloc[entry_idx])
                    entry_price = df['close'].iloc[entry_idx]
                    entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                    entries.append({
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
                lastFVG = -1
            elif bfvg_4h.iloc[i]:
                lastFVG = 1
            elif sfvg_4h.iloc[i]:
                lastFVG = -1

    return entries