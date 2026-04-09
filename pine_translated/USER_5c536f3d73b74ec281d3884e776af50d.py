import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Resample to 4H candles
    df = df.copy()
    df['ts_4h'] = (df['time'] // (4 * 60 * 60 * 1000)) * (4 * 60 * 60 * 1000)
    
    resampled = df.groupby('ts_4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'time': 'last'
    }).reset_index(drop=True)
    
    n = len(resampled)
    if n < 3:
        return []
    
    high = resampled['high']
    low = resampled['low']
    close = resampled['close']
    volume = resampled['volume']
    
    # Volume filter: volume[1] > ta.sma(volume, 9) * 1.5
    vol_ma = volume.rolling(9).mean()
    vol_filt = (volume.shift(1) > vol_ma * 1.5).fillna(False)
    
    # ATR filter (Wilder ATR): ta.atr(20) / 1.5
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, min_periods=20).mean()
    atr_filt = ((low - high.shift(2) > atr / 1.5) | (low.shift(2) - high > atr / 1.5)).fillna(False)
    
    # Trend filter: ta.sma(close, 54) > ta.sma(close, 54)[1]
    sma54 = close.rolling(54).mean()
    loc2 = (sma54 > sma54.shift(1)).fillna(False)
    
    # Bullish FVG: low_4h > high_4h[2] and volfilt1 and atrfilt1 and locfiltb1
    bullish_fvg = (low > high.shift(2)) & vol_filt & atr_filt & loc2
    
    # Bearish FVG: high_4h < low_4h[2] and volfilt1 and atrfilt1 and locfilts1
    bearish_fvg = (high < low.shift(2)) & vol_filt & atr_filt & ~loc2
    
    # Sharp Turn detection
    last_fvg = 0
    entries = []
    trade_num = 1
    
    for i in range(3, n):
        if bullish_fvg.iloc[i] and last_fvg == -1:
            ts = int(resampled['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
            last_fvg = 1
        elif bearish_fvg.iloc[i] and last_fvg == 1:
            ts = int(resampled['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
            last_fvg = -1
        elif bullish_fvg.iloc[i]:
            last_fvg = 1
        elif bearish_fvg.iloc[i]:
            last_fvg = -1
    
    return entries