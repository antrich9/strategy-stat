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
    
    # Initialize result list
    entries = []
    trade_num = 1
    
    # Ensure required columns exist
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        return entries
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Convert time to datetime for resampling
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('datetime', inplace=True)
    
    # Resample to 4H timeframe
    ohlc_4h = df[['open', 'high', 'low', 'close', 'volume']].resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Calculate 4H indicators
    # Volume filter: volume[1] > sma(volume, 9) * 1.5
    vol_sma_4h = ohlc_4h['volume'].rolling(9).mean()
    volfilt_4h = ohlc_4h['volume'].shift(1) > vol_sma_4h * 1.5
    
    # ATR filter (Wilder method): ATR(20) / 1.5
    high_low_4h = ohlc_4h['high'] - ohlc_4h['low']
    high_close_4h = np.abs(ohlc_4h['high'] - ohlc_4h['close'].shift(1))
    low_close_4h = np.abs(ohlc_4h['low'] - ohlc_4h['close'].shift(1))
    true_range_4h = pd.concat([high_low_4h, high_close_4h, low_close_4h], axis=1).max(axis=1)
    atr_4h = true_range_4h.ewm(alpha=1/20, adjust=False).mean() / 1.5
    
    # ATR filter condition
    atrfilt_4h = ((ohlc_4h['low'] - ohlc_4h['high'].shift(2) > atr_4h) | 
                  (ohlc_4h['low'].shift(2) - ohlc_4h['high'] > atr_4h))
    
    # Trend filter: SMA(54) direction
    sma_54_4h = ohlc_4h['close'].ewm(span=54, adjust=False).mean()
    locfiltb_4h = sma_54_4h > sma_54_4h.shift(1)
    locfilts_4h = ~locfiltb_4h
    
    # Bullish FVG: low > high[2] with all filters
    bfvg_4h = ((ohlc_4h['low'] > ohlc_4h['high'].shift(2)) & 
               volfilt_4h & atrfilt_4h & locfiltb_4h)
    
    # Bearish FVG: high < low[2] with all filters
    sfvg_4h = ((ohlc_4h['high'] < ohlc_4h['low'].shift(2)) & 
               volfilt_4h & atrfilt_4h & locfilts_4h)
    
    # Detect sharp turns (FVG direction change)
    # Track last FVG type
    last_fvg = 0  # 1 = bullish, -1 = bearish, 0 = none
    
    # Create a series to store sharp turn signals
    sharp_turn_bull = pd.Series(False, index=ohlc_4h.index)
    sharp_turn_bear = pd.Series(False, index=ohlc_4h.index)
    
    for i in range(len(ohlc_4h)):
        if i >= 2:
            if bfvg_4h.iloc[i] and last_fvg == -1:
                sharp_turn_bull.iloc[i] = True
                last_fvg = 1
            elif sfvg_4h.iloc[i] and last_fvg == 1:
                sharp_turn_bear.iloc[i] = True
                last_fvg = -1
            elif bfvg_4h.iloc[i]:
                last_fvg = 1
            elif sfvg_4h.iloc[i]:
                last_fvg = -1
    
    # Map 4H signals back to original 1H timeframe
    # Find the first 1H bar that falls within or after each 4H candle
    df_reset = df.reset_index()
    df_reset['4h_bucket'] = df_reset['datetime'].dt.floor('4H')
    
    # Create a mapping from 4H timestamp to 1H row indices
    entry_indices = []
    
    for idx in sharp_turn_bull[sharp_turn_bull].index:
        # Find the first 1H bar within this 4H candle
        mask = (df_reset['4h_bucket'] == idx) | ((df_reset['datetime'] > idx) & (df_reset['datetime'] <= idx + pd.Timedelta('4H')))
        matching_rows = df_reset[mask]
        if not matching_rows.empty:
            # Use the last row of the 4H candle (confirmed bar)
            entry_indices.append((matching_rows.index[-1], 'long'))
    
    for idx in sharp_turn_bear[sharp_turn_bear].index:
        mask = (df_reset['4h_bucket'] == idx) | ((df_reset['datetime'] > idx) & (df_reset['datetime'] <= idx + pd.Timedelta('4H')))
        matching_rows = df_reset[mask]
        if not matching_rows.empty:
            entry_indices.append((matching_rows.index[-1], 'short'))
    
    # Sort by index to maintain chronological order
    entry_indices.sort(key=lambda x: x[0])
    
    # Generate entry dicts
    for orig_idx, direction in entry_indices:
        entry_ts = int(df_reset['time'].iloc[orig_idx])
        entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        entry_price = float(df_reset['close'].iloc[orig_idx])
        
        entry = {
            'trade_num': trade_num,
            'direction': direction,
            'entry_ts': entry_ts,
            'entry_time': entry_time_str,
            'entry_price_guess': entry_price,
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': entry_price,
            'raw_price_b': entry_price
        }
        entries.append(entry)
        trade_num += 1
    
    return entries