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

    def ta_atr(high, low, close, length=20):
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/length, adjust=False).mean()
        return atr

    def is_in_london_window(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        dt_london = dt.tz_convert('Europe/London')
        hour = dt_london.hour
        minute = dt_london.minute
        in_morning = (hour == 7 and minute >= 45) or (hour in [8]) or (hour == 9 and minute <= 45)
        in_afternoon = (hour == 14 and minute >= 45) or (hour in [15]) or (hour == 16 and minute <= 45)
        return in_morning or in_afternoon

    close = df['close']
    open_series = df['open']
    high_series = df['high']
    low_series = df['low']
    volume = df['volume']

    close_shift1 = close.shift(1)
    open_shift1 = open_series.shift(1)
    high_shift1 = high_series.shift(1)
    low_shift1 = low_series.shift(1)
    high_shift2 = high_series.shift(2)
    low_shift2 = low_series.shift(2)
    vol_shift1 = volume.shift(1)
    vol_sma = volume.rolling(9).mean()
    atr = ta_atr(high_series, low_series, close, 20) / 1.5
    atr_filt = (low_series - high_shift2 > atr) | (low_series.shift(2) - high_series > atr)
    loc_series = close.rolling(54).mean()
    bull_trend = loc_series > loc_series.shift(1)
    bear_trend = ~bull_trend
    vol_filt = vol_shift1 > vol_sma * 1.5
    bull_filt = vol_filt & atr_filt & bull_trend
    bear_filt = vol_filt & atr_filt & bear_trend
    bull_fvg = (low_series > high_shift2) & bull_filt
    bear_fvg = (high_series < low_shift2) & bear_filt
    bull_ob = (close_shift1 <= open_shift1) & (close > open_series) & (close > high_shift1)
    bear_ob = (close_shift1 >= open_shift1) & (close < open_series) & (close < low_shift1)
    bull_entry = bull_fvg & bull_ob
    bear_entry = bear_fvg & bear_ob
    entries = []
    trade_num = 1

    for i in range(2, len(df)):
        ts = df['time'].iloc[i]
        if not is_in_london_window(ts):
            continue
        bull_cond = bull_entry.iloc[i]
        bear_cond = bear_entry.iloc[i]
        if pd.notna(bull_cond) and bull_cond:
            price = df['close'].iloc[i]
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(price),
                'raw_price_b': float(price)
            }
            entries.append(entry)
            trade_num += 1
        if pd.notna(bear_cond) and bear_cond:
            price = df['close'].iloc[i]
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(price),
                'raw_price_b': float(price)
            }
            entries.append(entry)
            trade_num += 1

    return entries