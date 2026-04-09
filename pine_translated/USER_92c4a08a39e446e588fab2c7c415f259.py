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
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    ts = df['time']

    emaF = 8
    emaS = 21
    voSLen = 5
    voLLen = 10
    zlF = 12
    zlS = 26
    zlSig = 9
    ttmsLen = 20

    baseline_fast = close.ewm(span=emaF, adjust=False).mean()
    baseline_slow = close.ewm(span=emaS, adjust=False).mean()

    ema_vol_short = volume.ewm(span=voSLen, adjust=False).mean()
    ema_vol_long = volume.ewm(span=voLLen, adjust=False).mean()
    vo = (ema_vol_short / ema_vol_long) - 1

    zlema1 = close.ewm(span=zlF, adjust=False).mean()
    zlema2 = close.ewm(span=zlS, adjust=False).mean()
    zlmacd = zlema1 - zlema2
    zlsignal = zlmacd.ewm(span=zlSig, adjust=False).mean()
    zlhist = zlmacd - zlsignal

    ma1 = close.ewm(span=ttmsLen, adjust=False).mean()
    ma2 = close.ewm(span=ttmsLen * 2, adjust=False).mean()
    ma3 = close.ewm(span=ttmsLen * 3, adjust=False).mean()
    ttms = (ma1 + ma2 + ma3) / 3
    ttms_ma = ttms.ewm(span=ttmsLen, adjust=False).mean()
    ttms_signal = (ttms - ttms_ma).ewm(span=5, adjust=False).mean()

    baseline_long = baseline_fast > baseline_slow
    baseline_short = baseline_fast < baseline_slow

    vo_long = vo > 0
    vo_short = vo < 0

    zl_long = zlhist > 0
    zl_short = zlhist < 0

    ttms_long = ttms_signal > 0
    ttms_short = ttms_signal < 0

    entries_long = baseline_long & vo_long & zl_long & ttms_long
    entries_short = baseline_short & vo_short & zl_short & ttms_short

    trade_num = 0
    entries = []

    for i in range(1, len(df)):
        if pd.isna(baseline_fast.iloc[i]) or pd.isna(baseline_slow.iloc[i]):
            continue
        if pd.isna(vo.iloc[i]):
            continue
        if pd.isna(zlhist.iloc[i]):
            continue
        if pd.isna(ttms_signal.iloc[i]):
            continue

        if entries_long.iloc[i]:
            trade_num += 1
            entry_ts = int(ts.iloc[i])
            entry_price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

        if entries_short.iloc[i]:
            trade_num += 1
            entry_ts = int(ts.iloc[i])
            entry_price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

    return entries