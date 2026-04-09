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
    # Helper: Wilder RSI
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Helper: Wilder ATR
    def wilder_atr(high, low, close, length):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr

    # Settings (default values from inputs)
    inp1 = False  # Volume Filter
    inp2 = False  # ATR Filter
    inp3 = False  # Trend Filter
    atr_length1 = 20
    lookback_bars = 12
    threshold = 0.0

    # Resample to 4H
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df_4h = df.set_index('time_dt').resample('240T').agg({
        'time': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna(subset=['high']).reset_index(drop=True)

    # 4H indicators
    high_4h1 = df_4h['high']
    low_4h1 = df_4h['low']
    close_4h1 = df_4h['close']
    volume_4h1 = df_4h['volume']

    # Volume filter
    volfilt1_4h = volume_4h1.rolling(9).mean() * 1.5
    volfilt1 = volume_4h1.shift(1) > volfilt1_4h.shift(1) if inp1 else pd.Series(True, index=volume_4h1.index)

    # ATR filter
    atr_4h = wilder_atr(high_4h1, low_4h1, close_4h1, atr_length1) / 1.5
    atrfilt1 = ((low_4h1 - high_4h1.shift(2) > atr_4h) | (low_4h1.shift(2) - high_4h1 > atr_4h)) if inp2 else pd.Series(True, index=low_4h1.index)

    # Trend filter
    loc1 = close_4h1.rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    locfiltb1 = loc21 if inp3 else pd.Series(True, index=loc1.index)
    locfilts1 = ~loc21 if inp3 else pd.Series(True, index=loc1.index)

    # Bullish/Bearish FVG on 4H
    bfvg1 = (low_4h1 > high_4h1.shift(2)) & volfilt1 & atrfilt1 & locfiltb1
    sfvg1 = (high_4h1 < low_4h1.shift(2)) & volfilt1 & atrfilt1 & locfilts1

    # Detect new 4H candles
    is_new_4h1 = df_4h.index.to_series().diff() != 0
    is_new_4h1.iloc[0] = True

    # Track last FVG type for sharp turn detection
    lastFVG = 0
    lastFVG_arr = np.zeros(len(df_4h), dtype=int)
    
    for i in range(1, len(df_4h)):
        if is_new_4h1.iloc[i]:
            if bfvg1.iloc[i] and lastFVG == -1:
                lastFVG = 1
            elif sfvg1.iloc[i] and lastFVG == 1:
                lastFVG = -1
            elif bfvg1.iloc[i]:
                lastFVG = 1
            elif sfvg1.iloc[i]:
                lastFVG = -1
        lastFVG_arr[i] = lastFVG

    df_4h['lastFVG'] = lastFVG_arr
    df_4h['sharp_turn_bull'] = bfvg1 & (df_4h['lastFVG'].shift(1) == -1) & is_new_4h1
    df_4h['sharp_turn_bear'] = sfvg1 & (df_4h['lastFVG'].shift(1) == 1) & is_new_4h1

    # Current timeframe FVG
    volfilt = df['volume'].shift(1) > df['volume'].rolling(9).mean().shift(1) * 1.5 if inp1 else pd.Series(True, index=df.index)
    atr2 = wilder_atr(df['high'], df['low'], df['close'], 20) / 1.5
    atrfilt = ((df['low'] - df['high'].shift(2) > atr2) | (df['low'].shift(2) - df['high'] > atr2)) if inp2 else pd.Series(True, index=df.index)
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2 if inp3 else pd.Series(True, index=df.index)
    locfilts = ~loc2 if inp3 else pd.Series(True, index=df.index)

    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts

    # BPR detection (Bullish)
    bear_fvg1 = (df['high'] < df['low'].shift(2)) & (df['close'].shift(1) < df['low'].shift(2))
    bull_fvg1 = (df['low'] > df['high'].shift(2)) & (df['close'].shift(1) > df['high'].shift(2))

    bull_since = np.zeros(len(df))
    for i in range(len(df)):
        count = 0
        for j in range(1, lookback_bars + 2):
            if i - j >= 0 and bear_fvg1.iloc[i - j]:
                count = j
                break
        bull_since[i] = count

    bull_cond_1 = bull_fvg1 & (bull_since <= lookback_bars)
    combined_low_bull = np.where(bull_cond_1, np.maximum(df['high'].values[np.maximum(0, np.array([bull_since[i] if i >= 0 else 0 for i in range(len(df))]).astype(int))], df['high'].shift(2).values), np.nan)
    combined_high_bull = np.where(bull_cond_1, np.minimum(df['low'].iloc[[max(0, int(bull_since[i])) for i in range(len(df))]].values.flatten(), df['low'].values), np.nan)
    
    bull_result = bull_cond_1 & (pd.Series(combined_high_bull, index=df.index) - pd.Series(combined_low_bull, index=df.index) >= threshold)

    # Detect new candles for BPR (similar to timeframe.change)
    is_new_bar = df.index.to_series().diff() != 0
    is_new_bar.iloc[0] = True

    # Merge 4H signals to current timeframe
    df['sharp_turn_bull'] = False
    df['sharp_turn_bear'] = False
    df['bull_result'] = bull_result

    # Map 4H signals to current timeframe using nearest timestamp
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df_4h['time_dt'] = pd.to_datetime(df_4h['time'], unit='s', utc=True)
    
    for idx, row in df_4h[df_4h['sharp_turn_bull'] | df_4h['sharp_turn_bear']].iterrows():
        closest_idx = (df['time_dt'] - row['time_dt']).abs().idxmin()
        if row['sharp_turn_bull']:
            df.loc[closest_idx, 'sharp_turn_bull'] = True
        if row['sharp_turn_bear']:
            df.loc[closest_idx, 'sharp_turn_bear'] = True

    # Build entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        entry_price = df['close'].iloc[i]
        
        if df['sharp_turn_bull'].iloc[i]:
            entry_time = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

        if df['sharp_turn_bear'].iloc[i]:
            entry_time = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

    return entries