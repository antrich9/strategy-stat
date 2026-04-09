import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Ensure required columns are present
    required_cols = {'time', 'open', 'high', 'low', 'close', 'volume'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Work on a copy to avoid side‑effects
    df = df.copy()

    # Basic price series
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']

    # Shifted values (proxy for dailyHigh, dailyLow, etc.)
    high_shift1 = high.shift(1)
    high_shift2 = high.shift(2)
    low_shift1 = low.shift(1)
    low_shift2 = low.shift(2)

    # ----- Volume filter (input inp1, default disabled) -----
    # Condition: volume[1] > SMA(volume,9) * 1.5
    vol_sma9 = volume.rolling(window=9).mean()
    vol_cond = (volume.shift(1) > vol_sma9 * 1.5).fillna(True)

    # ----- ATR filter (input inp2, default disabled) -----
    # True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder ATR (span = 20)
    atr20 = tr.ewm(span=20, adjust=False).mean()
    atr2 = atr20 / 1.5
    # Filter condition
    atrfilt_cond = ((low - high_shift2 > atr2) | (low_shift2 - high > atr2)).fillna(True)

    # ----- Trend filter (input inp3, default disabled) -----
    sma54 = close.rolling(window=54).mean()
    loc2 = sma54 > sma54.shift(1)
    locfiltb = loc2.fillna(True)     # for long entries
    locfilts = (~loc2).fillna(True)  # for short entries

    # ----- Entry condition series -----
    # Long: low > high of 2 bars back
    long_cond = (low > high_shift2).fillna(False) & vol_cond & atrfilt_cond & locfiltb
    # Short: high < low of 2 bars back
    short_cond = (high < low_shift2).fillna(False) & vol_cond & atrfilt_cond & locfilts

    # ----- Build entry list -----
    entries = []
    trade_num = 1

    for i in df.index:
        if not long_cond.loc[i] and not short_cond.loc[i]:
            continue

        ts = int(df.loc[i, 'time'])
        entry_price = float(df.loc[i, 'close'])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        if long_cond.loc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
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

        if short_cond.loc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
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

    return entries