import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Resample to 4H timeframe
    df_4h = df.copy()
    df_4h['datetime'] = pd.to_datetime(df_4h['time'], unit='s', utc=True)
    df_4h.set_index('datetime', inplace=True)
    
    ohlc_4h = df_4h.resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().copy()
    
    ohlc_4h.reset_index(inplace=True)
    ohlc_4h['time'] = ohlc_4h['datetime'].astype('int64') // 10**9
    
    n = len(ohlc_4h)
    
    # Volume filter: volume[1] > sma(volume, 9) * 1.5
    vol_sma = ohlc_4h['volume'].rolling(9).mean()
    volfilt = ohlc_4h['volume'].shift(1) > vol_sma * 1.5
    
    # ATR filter: (low - high[2] > atr_4h) or (low[2] - high > atr_4h)
    atr_length = 20
    tr_high = np.maximum(ohlc_4h['high'].values, ohlc_4h['close'].shift(1).values)
    tr_low = np.minimum(ohlc_4h['low'].values, ohlc_4h['close'].shift(1).values)
    tr = tr_high - tr_low
    atr_4h = pd.Series(tr).ewm(alpha=1/atr_length, adjust=False).mean()
    atrfilt = ((ohlc_4h['low'] - ohlc_4h['high'].shift(2) > atr_4h) | 
               (ohlc_4h['low'].shift(2) - ohlc_4h['high'] > atr_4h))
    
    # Trend filter: sma(close, 54) > sma(close, 54)[1]
    loc1 = ohlc_4h['close'].rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    locfiltb = loc21
    locfilts = ~loc21
    
    # Bullish FVG: low > high[2]
    bullish_fvg = (ohlc_4h['low'] > ohlc_4h['high'].shift(2)) & volfilt & atrfilt & locfiltb
    # Bearish FVG: high < low[2]
    bearish_fvg = (ohlc_4h['high'] < ohlc_4h['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    # Detect new 4H candle
    ohlc_4h['period'] = ohlc_4h['datetime'].dt.to_period('4H')
    is_new_4h = ([True] + (ohlc_4h['period'].values[1:] != ohlc_4h['period'].values[:-1]).tolist())
    ohlc_4h['is_new_4h'] = is_new_4h
    
    entries = []
    trade_num = 1
    last_fvg = 0
    
    for i in range(n):
        if ohlc_4h.iloc[i]['is_new_4h']:
            ts = int(ohlc_4h.iloc[i]['time'])
            price = float(ohlc_4h.iloc[i]['close'])
            
            # Bullish sharp turn: bullish FVG after bearish FVG
            if bullish_fvg.iloc[i] and last_fvg == -1:
                entry = {
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': price,
                    'raw_price_b': price
                }
                entries.append(entry)
                trade_num += 1
                last_fvg = 1
            # Bearish sharp turn: bearish FVG after bullish FVG
            elif bearish_fvg.iloc[i] and last_fvg == 1:
                entry = {
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': price,
                    'raw_price_b': price
                }
                entries.append(entry)
                trade_num += 1
                last_fvg = -1
            # Update last FVG state
            elif bullish_fvg.iloc[i]:
                last_fvg = 1
            elif bearish_fvg.iloc[i]:
                last_fvg = -1
    
    return entries