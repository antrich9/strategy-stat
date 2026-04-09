import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.
    """
    
    def calc_pivot_high(high, length):
        left = high.shift(1).rolling(length).max()
        right = high.shift(-length).rolling(length).max()
        return pd.concat([left, right], axis=1).max(axis=1)
    
    def calc_pivot_low(low, length):
        left = low.shift(1).rolling(length).min()
        right = low.shift(-length).rolling(length).min()
        return pd.concat([left, right], axis=1).min(axis=1)
    
    def calc_trend(high, low, close, length, display=True):
        ph = calc_pivot_high(high, length)
        pl = calc_pivot_low(low, length)
        trend = -1.0
        pivot_h = 0.0
        pivot_l = 0.0
        for i in range(length + 1, len(close)):
            if pd.isna(ph.iloc[i]) and pd.isna(pl.iloc[i]):
                continue
            if not pd.isna(ph.iloc[i]):
                pivot_h = ph.iloc[i]
            if not pd.isna(pl.iloc[i]):
                pivot_l = pl.iloc[i]
            co = (close.iloc[i] > pivot_h) and (close.iloc[i-1] <= ph.iloc[i-1] if not pd.isna(ph.iloc[i-1]) else True)
            cu = (close.iloc[i] < pivot_l) and (close.iloc[i-1] >= pl.iloc[i-1] if not pd.isna(pl.iloc[i-1]) else True)
            if co and trend == -1:
                trend = 1.0
            if cu and trend == 1:
                trend = -1.0
        return trend if display else 0.0
    
    trend1 = calc_trend(df['high'], df['low'], df['close'], 2)
    trend2 = calc_trend(df['high'], df['low'], df['close'], 5)
    trend3 = calc_trend(df['high'], df['low'], df['close'], 10)
    trend4 = calc_trend(df['high'], df['low'], df['close'], 15)
    trend5 = calc_trend(df['high'], df['low'], df['close'], 20)
    
    trend_avg = np.nanmean([trend1, trend2, trend3, trend4, trend5], axis=0)
    trend_series = pd.Series(trend_avg, index=df.index)
    
    bfvg = df['low'] > df['high'].shift(2)
    sfvg = df['high'] < df['low'].shift(2)
    
    hours = pd.to_datetime(df['time'], unit='s', utc=True).dt.hour
    is_morning = (hours >= 6) & (hours < 10)
    is_afternoon = (hours >= 14) & (hours < 18)
    in_trading_window = is_morning | is_afternoon
    
    entries = []
    trade_num = 1
    long_triggered = False
    short_triggered = False
    
    for i in range(1, len(df)):
        if not long_triggered and bfvg.iloc[i] and in_trading_window.iloc[i] and trend_series.iloc[i] >= 5:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            long_triggered = True
            trade_num += 1
        elif not short_triggered and sfvg.iloc[i] and in_trading_window.iloc[i] and trend_series.iloc[i] <= -5:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            short_triggered = True
            trade_num += 1
        if long_triggered and short_triggered:
            long_triggered = False
            short_triggered = False
    
    return entries