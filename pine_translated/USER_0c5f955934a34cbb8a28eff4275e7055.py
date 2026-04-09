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
    # Implement Wilder RSI manually
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Implement Wilder ATR manually
    def wilder_atr(high, low, close, length):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        return atr

    # Check if new 4H candle (timeframe change detection for 240min)
    def is_new_4h(time_series):
        # 4H = 240 minutes = 14400 seconds
        interval = 14400
        intervals_elapsed = (time_series / interval).astype(int)
        return intervals_elapsed.diff().fillna(0) > 0

    # Calculate 4H data using rolling window (last 16 bars of 15min = 4H)
    window_4h = 16
    
    high_4h = high_4h = df['high'].rolling(window_4h).max().shift(-window_4h + 1)
    low_4h = df['low'].rolling(window_4h).min().shift(-window_4h + 1)
    close_4h = df['close'].rolling(window_4h).mean().shift(-window_4h + 1)
    volume_4h = df['volume'].rolling(window_4h).sum().shift(-window_4h + 1)
    
    # Shift for [1] and [2] lookback on 4H data
    high_4h_1 = high_4h.shift(1)
    high_4h_2 = high_4h.shift(2)
    low_4h_1 = low_4h.shift(1)
    low_4h_2 = low_4h.shift(2)
    close_4h_1 = close_4h.shift(1)
    close_4h_2 = close_4h.shift(2)
    volume_4h_1 = volume_4h.shift(1)
    
    # Detect new 4H candle
    is_new_4h1 = is_new_4h(df['time'])
    
    # Volume Filter - using 4H data
    volfilt_series = volume_4h_1 > volume_4h.rolling(9).mean().shift(1) * 1.5
    
    # ATR Filter - using 4H data
    atr_length = 20
    atr_4h = wilder_atr(high_4h, low_4h, close_4h, atr_length) / 1.5
    atrfilt_series = (low_4h - high_4h_2 > atr_4h) | (low_4h_2 - high_4h > atr_4h)
    
    # Trend Filter - using 4H data
    loc = close_4h.rolling(54).mean()
    loc_1 = loc.shift(1)
    loc2_trend = loc > loc_1
    locfiltb = loc2_trend
    locfilts = ~loc2_trend
    
    # Identify Bullish and Bearish FVGs - using 4H data
    bfvg = (low_4h > high_4h_2) & volfilt_series & atrfilt_series & locfiltb
    sfvg = (high_4h < low_4h_2) & volfilt_series & atrfilt_series & locfilts
    
    # Track last FVG state (1=bullish, -1=bearish, 0=none)
    lastFVG_arr = np.zeros(len(df))
    lastFVG = 0
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        # Process only on new 4H candles to avoid duplicate signals
        if is_new_4h1.iloc[i]:
            # Check for NaN conditions
            if not (np.isnan(bfvg.iloc[i]) if isinstance(bfvg.iloc[i], float) else False):
                if bfvg.iloc[i]:
                    # Bullish Sharp Turn - lastFVG was -1 (bearish)
                    if lastFVG == -1:
                        entry_price = close_4h.iloc[i] if not np.isnan(close_4h.iloc[i]) else df['close'].iloc[i]
                        entries.append({
                            'trade_num': trade_num,
                            'direction': 'long',
                            'entry_ts': int(df['time'].iloc[i]),
                            'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                            'entry_price_guess': entry_price,
                            'exit_ts': 0,
                            'exit_time': '',
                            'exit_price_guess': 0.0,
                            'raw_price_a': entry_price,
                            'raw_price_b': entry_price
                        })
                        trade_num += 1
                        lastFVG = 1
                    elif lastFVG == 0:
                        lastFVG = 1
                elif not (np.isnan(sfvg.iloc[i]) if isinstance(sfvg.iloc[i], float) else False):
                    if sfvg.iloc[i]:
                        # Bearish Sharp Turn - lastFVG was 1 (bullish)
                        if lastFVG == 1:
                            entry_price = close_4h.iloc[i] if not np.isnan(close_4h.iloc[i]) else df['close'].iloc[i]
                            entries.append({
                                'trade_num': trade_num,
                                'direction': 'short',
                                'entry_ts': int(df['time'].iloc[i]),
                                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                                'entry_price_guess': entry_price,
                                'exit_ts': 0,
                                'exit_time': '',
                                'exit_price_guess': 0.0,
                                'raw_price_a': entry_price,
                                'raw_price_b': entry_price
                            })
                            trade_num += 1
                            lastFVG = -1
                        elif lastFVG == 0:
                            lastFVG = -1
    
    return entries