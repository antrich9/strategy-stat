import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    """
    open_prices = df['open']
    high_prices = df['high']
    low_prices = df['low']
    close_prices = df['close']
    volume = df['volume']
    timestamps = df['time']
    
    dtimestamps = pd.to_datetime(timestamps, unit='s', utc=True)
    hour = dtimestamps.dt.hour
    minute = dtimestamps.dt.minute
    time_window = ((hour == 7) & (minute >= 45)) | ((hour == 8) & (minute <= 45)) | \
                  ((hour == 14) & (minute >= 45)) | ((hour == 15) & (minute <= 45)) | \
                  (hour == 9) | (hour == 16)
    
    volfilt = volume.shift(1) > volume.rolling(9).mean() * 1.5
    tr1 = high_prices - low_prices
    tr2 = (high_prices - close_prices.shift(1)).abs()
    tr3 = (low_prices - close_prices.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atrfilt = (low_prices - high_prices.shift(2) > atr) | (low_prices.shift(2) - high_prices > atr)
    
    loc = close_prices.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfilt = loc2
    
    is_up = close_prices > open_prices
    is_down = close_prices < open_prices
    fvg_up = low_prices > high_prices.shift(2)
    fvg_down = high_prices < low_prices.shift(2)
    ob_up = is_down.shift(2) & is_up.shift(1) & (close_prices.shift(1) > high_prices.shift(2))
    ob_down = is_up.shift(2) & is_down.shift(1) & (close_prices.shift(1) < low_prices.shift(2))
    
    entry_long = fvg_up & ob_up & is_up & volfilt & atrfilt & locfilt
    entry_short = fvg_down & ob_down & is_down & volfilt & atrfilt & ~locfilt
    entry_long = entry_long & time_window
    entry_short = entry_short & time_window
    
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if entry_long.iloc[i]:
            ts = int(timestamps.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close_prices.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close_prices.iloc[i],
                'raw_price_b': close_prices.iloc[i]
            })
            trade_num += 1
        elif entry_short.iloc[i]:
            ts = int(timestamps.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close_prices.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close_prices.iloc[i],
                'raw_price_b': close_prices.iloc[i]
            })
            trade_num += 1
    return entries