import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    if len(df) < 3:
        return []

    vol = df['volume']
    high = df['high']
    low = df['low']
    close = df['close']

    # Wilder volume SMA (span=9)
    vol_sma = vol.ewm(span=9, adjust=False).mean()

    # Wilder ATR (length=20)
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()

    # Trend filter: 54 SMA
    loc = close.rolling(54).mean()
    loc_prev = loc.shift(1)
    loc2 = loc > loc_prev

    # FVG conditions on 1min data
    vol_filt = vol.shift(1) > vol_sma.shift(1) * 1.5
    atr_filt = ((low - high.shift(2) > atr / 1.5) | (low.shift(2) - high > atr.shift(2) / 1.5))
    trend_bull = loc2
    trend_bear = ~loc2

    bfvg = (low > high.shift(2)) & vol_filt & atr_filt & trend_bull
    sfvg = (high < low.shift(2)) & vol_filt & atr_filt & trend_bear

    entries = []
    trade_num = 1
    last_fvg = 0

    for i in range(3, len(df)):
        if pd.isna(close.iloc[i]) or pd.isna(trend_bull.iloc[i]) or pd.isna(bfvg.iloc[i]):
            continue

        curr_bfvg = bfvg.iloc[i]
        curr_sfvg = sfvg.iloc[i]

        if curr_bfvg and last_fvg == -1:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        elif curr_sfvg and last_fvg == 1:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1

        if curr_bfvg:
            last_fvg = 1
        elif curr_sfvg:
            last_fvg = -1

    return entries