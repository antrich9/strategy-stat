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
    if len(df) < 5:
        return []

    df = df.copy().reset_index(drop=True)
    ts = df['time']
    o = df['open']
    h = df['high']
    l = df['low']
    c = df['close']
    v = df['volume']

    daily = pd.DataFrame({
        'time': ts,
        'open': o,
        'high': h,
        'low': l,
        'close': c,
        'volume': v
    })
    daily['date'] = pd.to_datetime(daily['time'], unit='s', utc=True).dt.date
    daily_agg = daily.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()

    d_open = daily_agg['open'].values
    d_high = daily_agg['high'].values
    d_low = daily_agg['low'].values
    d_close = daily_agg['close'].values

    n = len(daily_agg)
    is_swing_high = np.zeros(n, dtype=bool)
    is_swing_low = np.zeros(n, dtype=bool)

    for i in range(4, n):
        if d_high[i-1] < d_high[i-2] and d_high[i-3] < d_high[i-2] and d_high[i-4] < d_high[i-2]:
            is_swing_high[i] = True
        if d_low[i-1] > d_low[i-2] and d_low[i-3] > d_low[i-2] and d_low[i-4] > d_low[i-2]:
            is_swing_low[i] = True

    swing_type = np.array(['none'] * n)
    for i in range(n):
        if is_swing_high[i]:
            swing_type[i] = 'dailyHigh'
        elif is_swing_low[i]:
            swing_type[i] = 'dailyLow'

    fvg_up = np.zeros(n, dtype=bool)
    fvg_down = np.zeros(n, dtype=bool)

    for i in range(2, n):
        if d_low[i] > d_high[i-2]:
            fvg_up[i] = True
        if d_high[i] < d_low[i-2]:
            fvg_down[i] = True

    vol_sma = pd.Series(v).rolling(9).mean()
    atr_raw = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = h.iloc[0] - l.iloc[0]
    for i in range(1, len(df)):
        tr[i] = max(h.iloc[i] - l.iloc[i], max(abs(h.iloc[i] - c.iloc[i-1]), abs(l.iloc[i] - c.iloc[i-1])))
    atr_smooth = np.zeros(n)
    atr_smooth[0] = tr[0]
    alpha = 1.0 / 20.0
    for i in range(1, n):
        atr_smooth[i] = (1 - alpha) * atr_smooth[i-1] + alpha * tr[i] if i < len(tr) else tr[i]
    for i in range(n):
        bar_idx = min(i * 1440 // (24 * 60) if 'minute' in str(df['time'].dtype) else i, len(atr_smooth) - 1)
    atr_val = np.zeros(len(df))
    for i in range(len(df)):
        day_idx = daily_agg['date'].searchsorted(pd.to_datetime(df['time'].iloc[i], unit='s', utc=True).date())[0] if isinstance(daily_agg['date'].searchsorted(pd.to_datetime(df['time'].iloc[i], unit='s', utc=True).date()), np.ndarray) else 0
        if i < len(atr_smooth):
            atr_val[i] = atr_smooth[i] / 1.5

    loc = pd.Series(c).rolling(54).mean()
    loc_slope = (loc > loc.shift(1)).astype(int)

    daily_idx_map = np.zeros(len(df), dtype=int) - 1
    for i in range(len(df)):
        bar_date = pd.to_datetime(df['time'].iloc[i], unit='s', utc=True).date()
        matches = np.where(daily_agg['date'] == bar_date)[0]
        if len(matches) > 0:
            daily_idx_map[i] = matches[0]

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if daily_idx_map[i] < 0:
            continue
        d_idx = daily_idx_map[i]
        if d_idx < 2:
            continue

        vol_ok = True
        atr_ok = True
        trend_bull = True
        trend_bear = True

        vol_filt = v.iloc[i] > vol_sma.iloc[i] * 1.5 if pd.notna(vol_sma.iloc[i]) else True
        vol_ok = vol_filt

        if d_idx > 2:
            low_diff = d_low[d_idx] - d_high[d_idx-2]
            high_diff = d_low[d_idx-2] - d_high[d_idx]
            atr_thresh = atr_smooth[d_idx] if d_idx < len(atr_smooth) else atr_smooth[-1]
            atr_ok = (low_diff > atr_thresh) or (high_diff > atr_thresh)

        if pd.notna(loc.iloc[i]) and pd.notna(loc.iloc[i-1]):
            trend_bull = loc.iloc[i] > loc.iloc[i-1]
            trend_bear = not trend_bull

        bfvg = fvg_up[d_idx] and vol_ok and atr_ok and trend_bull
        sfvg = fvg_down[d_idx] and vol_ok and atr_ok and trend_bear

        bull_swing = (swing_type[d_idx] == 'dailyLow') if d_idx < len(swing_type) else False
        bear_swing = (swing_type[d_idx] == 'dailyHigh') if d_idx < len(swing_type) else False

        if bfvg and bull_swing:
            entry_ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

        if sfvg and bear_swing:
            entry_ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries