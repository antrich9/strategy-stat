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
    
    fastLength = 50
    slowLength = 200
    
    close = df['close']
    high = df['high']
    low = df['low']
    open_arr = df['open']
    time_col = df['time']
    
    # Calculate EMAs
    fastEMA = close.ewm(span=fastLength, adjust=False).mean()
    slowEMA = close.ewm(span=slowLength, adjust=False).mean()
    
    # Calculate daily high/low for previous day computation
    df_temp = df.copy()
    df_temp['date'] = pd.to_datetime(df_temp['time'], unit='s', utc=True).dt.date
    daily_agg = df_temp.groupby('date').agg({'high': 'max', 'low': 'min'}).reset_index()
    daily_agg['prev_day_high'] = daily_agg['high'].shift(1)
    daily_agg['prev_day_low'] = daily_agg['low'].shift(1)
    daily_agg = daily_agg.dropna(subset=['prev_day_high', 'prev_day_low'])
    daily_agg = daily_agg[['date', 'prev_day_high', 'prev_day_low']]
    
    df_temp = df_temp.merge(daily_agg, on='date', how='left')
    df_temp['prevDayHigh'] = df_temp['prev_day_high']
    df_temp['prevDayLow'] = df_temp['prev_day_low']
    df_temp['prevDayHigh'] = df_temp['prevDayHigh'].ffill()
    df_temp['prevDayLow'] = df_temp['prevDayLow'].ffill()
    
    # Crossover and crossunder
    crossOver = (fastEMA > slowEMA) & (fastEMA.shift(1) <= slowEMA.shift(1))
    crossUnder = (fastEMA < slowEMA) & (fastEMA.shift(1) >= slowEMA.shift(1))
    
    # Previous day high/low sweep flags
    flagpdh = close > df_temp['prevDayHigh']
    flagpdl = close < df_temp['prevDayLow']
    
    # OB and FVG conditions
    def isUp(idx):
        return close.iloc[idx] > open_arr.iloc[idx]
    
    def isDown(idx):
        return close.iloc[idx] < open_arr.iloc[idx]
    
    def isObUp(idx):
        return isDown(idx + 1) and isUp(idx) and close.iloc[idx] > high.iloc[idx + 1]
    
    def isObDown(idx):
        return isUp(idx + 1) and isDown(idx) and close.iloc[idx] < low.iloc[idx + 1]
    
    def isFvgUp(idx):
        return low.iloc[idx] > high.iloc[idx + 2]
    
    def isFvgDown(idx):
        return high.iloc[idx] < low.iloc[idx + 2]
    
    # Compute OB and FVG as boolean series (shifted to align with current bar)
    obUp_series = pd.Series(False, index=df.index)
    obDown_series = pd.Series(False, index=df.index)
    fvgUp_series = pd.Series(False, index=df.index)
    fvgDown_series = pd.Series(False, index=df.index)
    
    for i in range(2, len(df) - 2):
        try:
            if isObUp(i):
                obUp_series.iloc[i] = True
            if isObDown(i):
                obDown_series.iloc[i] = True
            if isFvgUp(i):
                fvgUp_series.iloc[i] = True
            if isFvgDown(i):
                fvgDown_series.iloc[i] = True
        except:
            pass
    
    # Time filtering (0700-0959 UTC for longs, 1200-1459 UTC for shorts)
    dt = pd.to_datetime(df['time'], unit='s', utc=True)
    hour = dt.dt.hour
    
    long_time_filter = (hour >= 7) & (hour <= 9)
    short_time_filter = (hour >= 12) & (hour <= 14)
    
    # Entry conditions
    long_condition = crossOver & flagpdh & obUp_series & fvgUp_series & long_time_filter
    short_condition = crossUnder & flagpdl & obDown_series & fvgDown_series & short_time_filter
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_condition.iloc[i]:
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(time_col.iloc[i]),
                'entry_time': datetime.fromtimestamp(time_col.iloc[i] / 1000, tz=timezone.utc).isoformat() if time_col.iloc[i] > 1e12 else datetime.fromtimestamp(time_col.iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(time_col.iloc[i]),
                'entry_time': datetime.fromtimestamp(time_col.iloc[i] / 1000, tz=timezone.utc).isoformat() if time_col.iloc[i] > 1e12 else datetime.fromtimestamp(time_col.iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
    
    return entries