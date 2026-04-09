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
    df = df.copy()
    df['ts'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('ts', inplace=True)

    df['daily_high'] = df['high'].resample('D').max()
    df['daily_low'] = df['low'].resample('D').min()
    df['daily_close'] = df['close'].resample('D').last()

    df['daily_high_prev'] = df['daily_high'].shift(1)
    df['daily_low_prev'] = df['daily_low'].shift(1)
    df['daily_high_2prev'] = df['daily_high'].shift(2)
    df['daily_low_2prev'] = df['daily_low'].shift(2)

    vol_sma9 = df['volume'].rolling(9).mean()
    vol_filt = df['volume'].shift(1) > vol_sma9 * 1.5

    atr20 = df['high'] - df['low']
    tr1 = df['high'] - df['close'].shift(1)
    tr2 = df['close'].shift(1) - df['low']
    true_range = pd.concat([atr20, tr1, tr2], axis=1).max(axis=1)
    atr_211 = true_range.ewm(span=20, adjust=False).mean() / 1.5
    atr_filt = (df['daily_low_2prev'] - df['daily_high_2prev'] > atr_211) | (df['daily_low_2prev'] - df['daily_high_prev'] > atr_211)

    sma54 = df['close'].rolling(54).mean()
    loc211 = sma54 > sma54.shift(1)
    loc_filt_bull = loc211
    loc_filt_bear = ~loc211

    is_swing_high = (df['daily_high_prev'] < df['daily_high_2prev']) & \
                    (df['daily_high'] < df['daily_high_2prev']) & \
                    (df['daily_high'].shift(1) < df['daily_high_2prev']) & \
                    (df['daily_high'].shift(2) < df['daily_high_2prev'])
    is_swing_low = (df['daily_low_prev'] > df['daily_low_2prev']) & \
                   (df['daily_low'] > df['daily_low_2prev']) & \
                   (df['daily_low'].shift(1) > df['daily_low_2prev']) & \
                   (df['daily_low'].shift(2) > df['daily_low_2prev'])

    last_swing_type = pd.Series(index=df.index, dtype='object')
    for i in range(len(df)):
        if is_swing_high.iloc[i]:
            last_swing_type.iloc[i] = "dailyHigh"
        elif is_swing_low.iloc[i]:
            last_swing_type.iloc[i] = "dailyLow"

    last_swing_type = last_swing_type.ffill()

    bfvg = (df['daily_low'] > df['daily_high_2prev']) & vol_filt & atr_filt & loc_filt_bull
    sfvg = (df['daily_high_prev'] < df['daily_low_2prev']) & vol_filt & atr_filt & loc_filt_bear

    bullish_fvg = bfvg & (last_swing_type == "dailyLow")
    bearish_fvg = sfvg & (last_swing_type == "dailyHigh")

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if i < 4:
            continue
        if pd.isna(df['daily_high_2prev'].iloc[i]) or pd.isna(df['daily_low_2prev'].iloc[i]):
            continue

        if bullish_fvg.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1

        if bearish_fvg.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1

    return entries