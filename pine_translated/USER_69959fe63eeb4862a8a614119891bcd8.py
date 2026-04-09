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
    
    # Aggregate to 4H
    df_4h = df.copy()
    df_4h['time_dt'] = pd.to_datetime(df_4h['time'], unit='s', utc=True)
    df_4h.set_index('time_dt', inplace=True)
    
    # Resample to 4H
    ohlc_4h = df_4h[['open', 'high', 'low', 'close']].resample('240T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    volume_4h = df_4h['volume'].resample('240T').sum()
    
    ohlc_4h['volume'] = volume_4h
    ohlc_4h.reset_index(inplace=True)
    ohlc_4h['time'] = ohlc_4h['time_dt'].astype(np.int64) // 10**9
    
    # Volume Filter
    vol_sma = ohlc_4h['volume'].rolling(9).mean()
    vol_filt = ohlc_4h['volume'] > vol_sma * 1.5
    
    # ATR Filter (20-period Wilder ATR on 4H)
    high_low = ohlc_4h['high'] - ohlc_4h['low']
    high_close = np.abs(ohlc_4h['high'] - ohlc_4h['close'].shift())
    low_close = np.abs(ohlc_4h['low'] - ohlc_4h['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/20, adjust=False).mean() / 1.5
    atr_filt = (ohlc_4h['low'] - ohlc_4h['high'].shift(2) > atr) | (ohlc_4h['low'].shift(2) - ohlc_4h['high'] > atr)
    
    # Trend Filter (54-period SMA on 4H close)
    sma54 = ohlc_4h['close'].rolling(54).mean()
    trend_up = sma54 > sma54.shift()
    trend_filt_long = trend_up
    trend_filt_short = ~trend_up
    
    # Bullish FVG: low > high[2] and filters
    bfvg = (ohlc_4h['low'] > ohlc_4h['high'].shift(2)) & vol_filt & atr_filt & trend_filt_long
    
    # Bearish FVG: high < low[2] and filters
    sfvg = (ohlc_4h['high'] < ohlc_4h['low'].shift(2)) & vol_filt & atr_filt & trend_filt_short
    
    # Generate entries on sharp reversals
    entries = []
    last_fvg = 0
    trade_num = 1
    
    for i in range(1, len(ohlc_4h)):
        if last_fvg == -1 and bfvg.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ohlc_4h['time'].iloc[i],
                'entry_time': datetime.fromtimestamp(ohlc_4h['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price': ohlc_4h['close'].iloc[i],
                'raw_price_a': ohlc_4h['low'].iloc[i] - ohlc_4h['high'].iloc[i-2],
                'raw_price_b': ohlc_4h['close'].iloc[i]
            })
            trade_num += 1
            last_fvg = 1
        elif last_fvg == 1 and sfvg.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ohlc_4h['time'].iloc[i],
                'entry_time': datetime.fromtimestamp(ohlc_4h['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price': ohlc_4h['close'].iloc[i],
                'raw_price_a': ohlc_4h['high'].iloc[i] - ohlc_4h['low'].iloc[i-2],
                'raw_price_b': ohlc_4h['close'].iloc[i]
            })
            trade_num += 1
            last_fvg = -1
        elif bfvg.iloc[i]:
            last_fvg = 1
        elif sfvg.iloc[i]:
            last_fvg = -1
    
    return entries