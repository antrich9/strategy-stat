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
    
    # Resample to 4H timeframe
    df_4h = df.copy()
    df_4h['datetime'] = pd.to_datetime(df_4h['time'], unit='s', utc=True)
    df_4h.set_index('datetime', inplace=True)
    
    # Resample to 4H (240 min)
    ohlc_4h = df_4h[['open', 'high', 'low', 'close', 'volume']].resample('240T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    ohlc_4h = ohlc_4h.reset_index()
    ohlc_4h['time'] = ohlc_4h['datetime'].apply(lambda x: int(x.timestamp()))
    
    # Detect new 4H candle
    is_new_4h = np.concatenate([[True], ohlc_4h['time'].values[1:] != ohlc_4h['time'].values[:-1]])
    
    # Calculate Volume SMA (9 period) for 4H
    vol_sma_4h = ohlc_4h['volume'].rolling(9).mean()
    
    # Calculate ATR (20 period, Wilder's method)
    high = ohlc_4h['high'].values
    low = ohlc_4h['low'].values
    close_4h = ohlc_4h['close'].values
    
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close_4h[:-1]), np.abs(low[1:] - close_4h[:-1])))
    tr = np.concatenate([[np.nan], tr])
    
    atr_4h = np.zeros(len(ohlc_4h))
    atr_4h[0] = np.nan
    multiplier = 1.0 / 20
    for i in range(1, len(ohlc_4h)):
        if i == 1:
            atr_4h[i] = tr[i]
        else:
            atr_4h[i] = (atr_4h[i-1] * (1 - multiplier)) + (tr[i] * multiplier)
    
    # Calculate SMA of close (54 period) for trend filter
    sma_close_54 = ohlc_4h['close'].rolling(54).mean()
    
    # Trend filter
    loc21 = sma_close_54 > sma_close_54.shift(1)
    
    # FVG detection on 4H
    low_4h = ohlc_4h['low'].values
    high_4h = ohlc_4h['high'].values
    
    # Bullish FVG: low > high[2]
    bfvg1 = np.concatenate([[False, False], (low_4h[2:] > high_4h[:-2])])
    bfvg1[:2] = False
    
    # Bearish FVG: high < low[2]
    sfvg1 = np.concatenate([[False, False], (high_4h[2:] < low_4h[:-2])])
    sfvg1[:2] = False
    
    # Volume Filter: volume[1] > sma(volume, 9) * 1.5
    vol_filter = np.concatenate([[True], (ohlc_4h['volume'].values[1:] > vol_sma_4h.values[1:] * 1.5)])
    vol_filter[0] = True
    
    # ATR Filter: (low - high[2] > atr/1.5) or (low[2] - high > atr/1.5)
    atr_adjusted = atr_4h / 1.5
    atr_filter = np.concatenate([[True], (
        ((low_4h[1:] - high_4h[:-1]) > atr_adjusted[1:]) | 
        ((low_4h[:-1] - high_4h[1:]) > atr_adjusted[1:])
    )])
    atr_filter[0] = True
    
    # Trend filter
    locfiltb1 = loc21.copy()
    locfilts1 = ~loc21.copy()
    
    # Combined FVG conditions
    bfvg_combined = bfvg1 & vol_filter & atr_filter & locfiltb1
    sfvg_combined = sfvg1 & vol_filter & atr_filter & locfilts1
    
    # London time window check (07:45 to 17:45 UTC)
    times = pd.to_datetime(ohlc_4h['time'], unit='s', utc=True)
    hours = times.hour + times.minute / 60
    in_time_window = (hours >= 7.75) & (hours < 17.75)
    
    # Track last FVG type
    lastFVG = 0
    entries = []
    trade_num = 1
    
    for i in range(len(ohlc_4h)):
        if is_new_4h[i] and in_time_window.iloc[i]:
            if bfvg_combined[i] and lastFVG == -1:
                # Bullish Sharp Turn - LONG entry
                entry_ts = int(ohlc_4h['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(ohlc_4h['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(ohlc_4h['close'].iloc[i]),
                    'raw_price_b': float(ohlc_4h['close'].iloc[i])
                })
                trade_num += 1
                lastFVG = 1
            elif sfvg_combined[i] and lastFVG == 1:
                # Bearish Sharp Turn - SHORT entry
                entry_ts = int(ohlc_4h['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(ohlc_4h['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(ohlc_4h['close'].iloc[i]),
                    'raw_price_b': float(ohlc_4h['close'].iloc[i])
                })
                trade_num += 1
                lastFVG = -1
            elif bfvg_combined[i]:
                lastFVG = 1
            elif sfvg_combined[i]:
                lastFVG = -1
    
    return entries