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
    if len(df) < 20:
        return []

    # Resample to 4H
    df_4h = df.copy()
    df_4h['time'] = pd.to_datetime(df_4h['time'], unit='s', utc=True)
    df_4h.set_index('time', inplace=True)
    
    # Resample to 4H using first/last for OHLC and sum for volume
    resampled = df_4h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Reset index to have time as column
    resampled = resampled.reset_index()
    resampled['time'] = resampled['time'].apply(lambda x: int(x.timestamp()))
    
    # Helper: Wilder RSI (not directly used but available)
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
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
    
    # Calculate 4H indicators
    high_4h = resampled['high']
    low_4h = resampled['low']
    close_4h = resampled['close']
    volume_4h = resampled['volume']
    
    # Volume Filter
    vol_sma_4h = volume_4h.rolling(9).mean()
    volfilt1 = vol_sma_4h * 1.5
    
    # ATR Filter (4H)
    atr_length1 = 20
    atr_4h = wilder_atr(high_4h, low_4h, close_4h, atr_length1)
    atr_4h_adj = atr_4h / 1.5
    atr_filt_condition = (low_4h - high_4h.shift(2) > atr_4h_adj) | (low_4h.shift(2) - high_4h > atr_4h_adj)
    
    # Trend Filter (4H)
    loc1 = close_4h.rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    
    # Bullish FVG on 4H
    bfvg1 = (low_4h > high_4h.shift(2)) & (volume_4h.shift(1) > volfilt1) & atr_filt_condition & loc21
    
    # Bearish FVG on 4H
    sfvg1 = (high_4h < low_4h.shift(2)) & (volume_4h.shift(1) > volfilt1) & atr_filt_condition & (~loc21)
    
    # Detect new 4H candles (compare timestamps)
    is_new_4h = pd.Series([True] + [resampled['time'].iloc[i] >= resampled['time'].iloc[i-1] + 14400 for i in range(1, len(resampled))], index=resampled.index)
    
    # Sharp turn detection: only on new 4H candles
    entries = []
    trade_num = 1
    lastFVG = 0  # 1=bullish, -1=bearish, 0=none
    
    for i in range(2, len(resampled)):
        if not is_new_4h.iloc[i]:
            continue
        
        bfvg = bfvg1.iloc[i]
        sfvg = sfvg1.iloc[i]
        
        # Sharp Turn Long: Bullish FVG after Bearish FVG
        if bfvg and lastFVG == -1:
            entry_ts = resampled['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(resampled['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(resampled['close'].iloc[i]),
                'raw_price_b': float(resampled['close'].iloc[i])
            })
            trade_num += 1
            lastFVG = 1
        # Sharp Turn Short: Bearish FVG after Bullish FVG
        elif sfvg and lastFVG == 1:
            entry_ts = resampled['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(resampled['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(resampled['close'].iloc[i]),
                'raw_price_b': float(resampled['close'].iloc[i])
            })
            trade_num += 1
            lastFVG = -1
        elif bfvg:
            lastFVG = 1
        elif sfvg:
            lastFVG = -1
    
    return entries