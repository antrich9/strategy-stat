import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # default inputs
    trend_filter = True
    vol_ma_period = 20
    vol_multiplier = 1.5
    require_vol_surge = True
    max_trades_per_day = 2

    close = df['close']
    volume = df['volume']

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    vol_ma = volume.rolling(vol_ma_period).mean()

    uptrend = sma50 > sma200
    downtrend = sma50 < sma200
    if trend_filter:
        trend_valid = uptrend | downtrend
    else:
        trend_valid = pd.Series(True, index=df.index)

    volume_surge = volume > vol_ma * vol_multiplier
    if require_vol_surge:
        vol_confirmed = volume_surge
    else:
        vol_confirmed = pd.Series(True, index=df.index)

    long_cond = uptrend & vol_confirmed & trend_valid & (close > sma20)
    short_cond = downtrend & vol_confirmed & trend_valid & (close < sma20)

    entries = []
    trade_num = 1
    trades_today_count = 0
    prev_day = None

    for i in df.index:
        if pd.isna(sma20.loc[i]) or pd.isna(sma50.loc[i]) or pd.isna(sma200.loc[i]) or pd.isna(vol_ma.loc[i]):
            continue

        ts = int(df.loc[i, 'time'])
        day = ts // 86400
        if prev_day is not None and day != prev_day:
            trades_today_count = 0
        prev_day = day

        if trades_today_count >= max_trades_per_day:
            continue

        if long_cond.loc[i]:
            direction = 'long'
        elif short_cond.loc[i]:
            direction = 'short'
        else:
            continue

        entry_price = float(df.loc[i, 'close'])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entries.append({
            'trade_num': trade_num,
            'direction': direction,
            'entry_ts': ts,
            'entry_time': entry_time,
            'entry_price_guess': entry_price,
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': entry_price,
            'raw_price_b': entry_price
        })
        trade_num += 1
        trades_today_count += 1

    return entries