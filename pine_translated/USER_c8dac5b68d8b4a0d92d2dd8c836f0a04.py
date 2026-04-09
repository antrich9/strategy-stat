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
    entries = []
    trade_num = 1

    if len(df) < 5:
        return entries

    # Convert time to datetime for time-based filtering
    dt = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc))
    hour = dt.dt.hour
    minute = dt.dt.minute
    month = dt.dt.month
    day = dt.dt.day
    dayofweek = dt.dt.dayofweek

    # Trading windows: 07:00-10:59 and 15:00-16:59 UTC
    in_window_1 = (hour >= 7) & (hour < 11)
    in_window_2 = (hour >= 15) & (hour < 17)
    in_trading_window = in_window_1 | in_window_2

    # Volume filter
    vol_sma = df['volume'].rolling(window=9).mean()
    vol_filter = df['volume'].shift(1) > vol_sma * 1.5

    # ATR filter (Wilder)
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(np.abs(df['high'] - df['close'].shift(1)),
                               np.abs(df['low'] - df['close'].shift(1))))
    atr = pd.Series(index=df.index, dtype=float)
    atr.iloc[0] = tr.iloc[0]
    for i in range(1, len(df)):
        atr.iloc[i] = (atr.iloc[i-1] * 19 + tr.iloc[i]) / 20
    atr_val = atr / 1.5
    low_diff = df['low'] - df['high'].shift(2)
    high_diff = df['low'].shift(2) - df['high']
    atr_filter = (low_diff > atr_val) | (high_diff > atr_val)

    # Trend filter
    loc = df['close'].rolling(window=54).mean()
    loc2 = loc > loc.shift(1)
    loc_filter_bull = loc2
    loc_filter_bear = ~loc2

    # Bullish and bearish FVG conditions
    bfvg = (df['low'] > df['high'].shift(2)) & vol_filter & atr_filter & loc_filter_bull
    sfvg = (df['high'] < df['low'].shift(2)) & vol_filter & atr_filter & loc_filter_bear

    # OB and FVG stacked conditions
    # isObUp at bar i: isDown(i-1) and isUp(i) and close[i] > high[i-1]
    # isFvgUp at bar i: low[i] > high[i+2] (fvgUp checks forward, so at bar i, check low[i] vs high[i-2] from original)
    # Adjusted for shift: fvgUp[i] = df['low'].iloc[i] > df['high'].iloc[i-2]
    bullish_ob = (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open']) & (df['close'] > df['high'].shift(1))
    bearish_ob = (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open']) & (df['close'] < df['low'].shift(1))
    bullish_fvg = df['low'] > df['high'].shift(2)
    bearish_fvg = df['high'] < df['low'].shift(2)

    bullish_stacked = bullish_ob & bullish_fvg
    bearish_stacked = bearish_ob & bearish_fvg

    for i in range(2, len(df)):
        if pd.isna(bfvg.iloc[i]) or pd.isna(sfvg.iloc[i]):
            continue
        if in_trading_window.iloc[i]:
            if bullish_stacked.iloc[i] and not pd.isna(bullish_stacked.iloc[i]):
                entry_price = df['close'].iloc[i]
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
            elif bearish_stacked.iloc[i] and not pd.isna(bearish_stacked.iloc[i]):
                entry_price = df['close'].iloc[i]
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1

    return entries