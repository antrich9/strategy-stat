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

    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']

    # Volume Filter
    vol_sma = volume.rolling(9).mean()
    vol_filt = volume.shift(1) > vol_sma * 1.5

    # ATR Filter (ATR(20) / 1.5) - Wilder's smoothed
    tr1 = high - low.shift(1)
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0/14, adjust=False).mean()
    atr_filt = (low.shift(2) - high > atr / 1.5) | (low - high.shift(2) > atr / 1.5)

    # Trend Filter (SMA 54)
    sma54 = close.rolling(54).mean()
    loc2 = sma54 > sma54.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    # FVG Detection
    bfvg = low > high.shift(2)
    sfvg = high < low.shift(2)

    # Track last FVG state
    lastFVG_series = pd.Series(0, index=df.index)
    lastFVG = 0

    for i in range(5, len(df)):
        if bfvg.iloc[i]:
            lastFVG = 1
        elif sfvg.iloc[i]:
            lastFVG = -1
        lastFVG_series.iloc[i] = lastFVG

    # 240 timeframe filter (using local close SMA(50))
    tf_sma = close.rolling(50).mean()
    bullish_tf_cond = close > tf_sma
    bearish_tf_cond = close < tf_sma

    # Entry conditions
    long_condition = vol_filt & atr_filt & locfiltb & bfvg & (lastFVG_series == -1) & bullish_tf_cond
    short_condition = vol_filt & atr_filt & locfilts & sfvg & (lastFVG_series == 1) & bearish_tf_cond

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(5, len(df)):
        if pd.isna(vol_filt.iloc[i]) or pd.isna(atr_filt.iloc[i]) or pd.isna(locfiltb.iloc[i]):
            continue

        if long_condition.iloc[i]:
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
        elif short_condition.iloc[i]:
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

    return entries