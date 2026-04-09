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
    
    # Volume Filter: volfilt = volume[1] > ta.sma(volume, 9) * 1.5
    sma9_vol = df['volume'].rolling(9).mean()
    volfilt = df['volume'] > sma9_vol * 1.5
    
    # ATR (Wilder) for ATR filter: atr = ta.atr(20) / 1.5
    high = df['high']
    low = df['low']
    close = df['close']
    
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_raw = np.zeros(len(df))
    atr_raw[0] = tr.iloc[0]
    alpha = 1.0 / 20.0
    for i in range(1, len(df)):
        atr_raw[i] = (atr_raw[i-1] * (1 - alpha) + tr.iloc[i] * alpha)
    atr_val = pd.Series(atr_raw, index=df.index)
    atrfilt = ((low - high.shift(2)) > (atr_val / 1.5)) | ((low.shift(2) - high) > (atr_val / 1.5))
    
    # Trend Filter: loc = ta.sma(close, 54), loc2 = loc > loc[1]
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # Bullish FVG: low > high[2] and volfilt and atrfilt and locfiltb
    bull_fvg = (low > high.shift(2)) & volfilt & atrfilt & locfiltb
    
    # Bearish FVG: high < low[2] and volfilt and atrfilt and locfilts
    bear_fvg = (high < low.shift(2)) & volfilt & atrfilt & locfilts
    
    # Time window check (London morning and afternoon sessions)
    ts = pd.to_datetime(df['time'], unit='s', utc=True)
    london_time = ts.dt.tz_convert('Europe/London')
    hours = london_time.dt.hour
    minutes = london_time.dt.minute
    minute_of_day = hours * 60 + minutes
    
    in_morning = (minute_of_day >= 405) & (minute_of_day < 525)
    in_afternoon = (minute_of_day >= 885) & (minute_of_day < 945)
    in_trading_window = in_morning | in_afternoon
    
    # Sharp turn detection: FVG in opposite direction of previous bar's FVG
    bull_sharp_turn = bull_fvg & bear_fvg.shift(1)
    bear_sharp_turn = bear_fvg & bull_fvg.shift(1)
    
    entries = []
    trade_num = 0
    
    for i in range(1, len(df)):
        long_signal = bull_sharp_turn.iloc[i] and not in_trading_window.iloc[i]
        short_signal = bear_sharp_turn.iloc[i] and not in_trading_window.iloc[i]
        
        if long_signal:
            trade_num += 1
            ts_val = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts_val,
                'entry_time': datetime.fromtimestamp(ts_val, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            long_signal = False
        
        if short_signal:
            trade_num += 1
            ts_val = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts_val,
                'entry_time': datetime.fromtimestamp(ts_val, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            short_signal = False
    
    return entries