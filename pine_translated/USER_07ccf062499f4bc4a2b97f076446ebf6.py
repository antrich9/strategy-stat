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
    
    o = df['open']
    h = df['high']
    l = df['low']
    c = df['close']
    v = df['volume']
    t = df['time']
    
    # Detect if bar is up or down
    is_up = c > o
    is_down = c < o
    
    # Detect bullish OB at index i: close[i] > high[i-1], with isDown(i+1) and isUp(i)
    # Using shift: OB at i when close[i] > high[i-1] AND close[i+1] < open[i+1] AND close[i] > open[i]
    ob_up_series = is_down.shift(-1) & is_up & (c > h.shift(1))
    
    # Detect bearish OB at index i: close[i] < low[i-1], with isUp(i+1) and isDown(i)
    ob_down_series = is_up.shift(-1) & is_down & (c < l.shift(1))
    
    # Bullish FVG: low > high[2]
    bfvg_series = l > h.shift(2)
    
    # Bearish FVG: high < low[2]
    sfvg_series = h < l.shift(2)
    
    # Volume filter: volume[1] > ta.sma(volume, 9) * 1.5
    vol_sma = v.rolling(9).mean()
    volfilt_series = v.shift(1) > vol_sma * 1.5
    
    # ATR filter: ta.atr(20) / 1.5
    tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atrfilt_series = (l - h.shift(2) > atr/1.5) | (l.shift(2) - h > atr/1.5)
    
    # Trend filter: ta.sma(close, 54)
    loc = c.ewm(span=54, adjust=False).mean()
    locfiltb_series = loc > loc.shift(1)
    locfilts_series = loc < loc.shift(1)
    
    # Combined bullish and bearish conditions
    bull_cond = bfvg_series & volfilt_series & atrfilt_series & locfiltb_series & ob_up_series.shift(1)
    bear_cond = sfvg_series & volfilt_series & atrfilt_series & locfilts_series & ob_down_series.shift(1)
    
    # Time window check (UTC+1 adjusted hours)
    dt = pd.to_datetime(t, unit='s', utc=True)
    hours = dt.dt.hour
    minutes = dt.dt.minute
    
    # Window 1: 07:00 - 10:59
    in_window_1 = (hours >= 7) & ((hours < 10) | ((hours == 10) & (minutes <= 59)))
    # Window 2: 15:00 - 16:59
    in_window_2 = (hours >= 15) & ((hours < 16) | ((hours == 16) & (minutes <= 59)))
    in_trading_window = in_window_1 | in_window_2
    
    # Final entry conditions
    long_cond = bull_cond & in_trading_window
    short_cond = bear_cond & in_trading_window
    
    # Detect crossover (entry trigger)
    long_entry = long_cond & ~long_cond.shift(1).fillna(False)
    short_entry = short_cond & ~short_cond.shift(1).fillna(False)
    
    # Build results
    entries = []
    trade_num = 1
    
    for i in range(2, len(df)):
        if i < len(long_entry) and i < len(short_entry):
            if long_entry.iloc[i]:
                ts = int(t.iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(c.iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(c.iloc[i]),
                    'raw_price_b': float(c.iloc[i])
                })
                trade_num += 1
            elif short_entry.iloc[i]:
                ts = int(t.iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(c.iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(c.iloc[i]),
                    'raw_price_b': float(c.iloc[i])
                })
                trade_num += 1
    
    return entries