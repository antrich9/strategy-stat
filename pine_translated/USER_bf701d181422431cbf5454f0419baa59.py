import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    dt = pd.to_datetime(df['time'], unit='s', utc=True)
    month = dt.dt.month
    day = dt.dt.day
    dayofweek = dt.dt.dayofweek

    is_dst = ((month == 3) & (day >= 25) & (dayofweek == 6)) | (month > 3) & ((month == 10) & (day < 7) & (dayofweek == 6) | (month < 10))
    adjusted_time_ms = df['time'] * 1000 + np.where(is_dst, 7200000, 3600000)
    adjusted_dt = pd.to_datetime(adjusted_time_ms, unit='ms', utc=True)
    adjusted_hour = adjusted_dt.dt.hour
    adjusted_minute = adjusted_dt.dt.minute

    in_trading_window = ((adjusted_hour >= 7) & (adjusted_hour < 10)) | ((adjusted_hour == 10) & (adjusted_minute <= 59)) | ((adjusted_hour >= 15) & (adjusted_hour < 16)) | ((adjusted_hour == 16) & (adjusted_minute <= 59))

    isUp = df['close'] > df['open']
    isDown = df['close'] < df['open']
    obUp = isDown.shift(1) & isUp & (df['close'] > df['high'].shift(1))
    obDown = isUp.shift(1) & isDown & (df['close'] < df['low'].shift(1))
    fvgUp = df['low'] > df['high'].shift(2)
    fvgDown = df['high'] < df['low'].shift(2)

    volfilt = (df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5)

    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atrfilt = ((df['low'] - df['high'].shift(2) > atr / 1.5) | (df['low'].shift(2) - df['high'] > atr / 1.5))

    loc = df['close'].ewm(span=54, adjust=False).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    bull_fvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    bear_fvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts

    bull_cond = obUp.shift(1) & fvgUp.shift(1) & in_trading_window & bull_fvg
    bear_cond = obDown.shift(1) & fvgDown.shift(1) & in_trading_window & bear_fvg

    skip_first = 55
    entries = []
    trade_num = 1

    for i in range(skip_first, len(df)):
        if bull_cond.iloc[i] or bear_cond.iloc[i]:
            direction = 'long' if bull_cond.iloc[i] else 'short'
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1

    return entries