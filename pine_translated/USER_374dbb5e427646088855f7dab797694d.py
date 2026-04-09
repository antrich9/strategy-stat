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
    if len(df) < 20:
        return []

    entries = []
    trade_num = 1

    # Extract hour and minute from timestamps (assuming UTC data)
    hours = pd.to_datetime(df['time'], unit='ms', utc=True).dt.hour
    minutes = pd.to_datetime(df['time'], unit='ms', utc=True).dt.minute

    # 4H period identification (0, 4, 8, 12, 16, 20)
    period_4h = (hours // 4) * 4
    is_new_4h = (period_4h != period_4h.shift(1)) & period_4h.notna()

    # Create 4H aggregated data
    group_4h = (hours // 4) * 4
    df_4h = df.groupby(group_4h, sort=False).agg({
        'time': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index(drop=True)

    # FVG conditions using 4H data
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    close_4h = df_4h['close']
    volume_4h = df_4h['volume']

    # Volume filter: volume_4h[1] > ta.sma(volume_4h, 9) * 1.5
    vol_sma = volume_4h.rolling(9, min_periods=9).mean()
    vol_filt_4h = volume_4h.shift(1) > vol_sma.shift(1) * 1.5

    # ATR filter: (low_4h - high_4h[2] > atr_4h) or (low_4h[2] - high_4h > atr_4h)
    tr1 = high_4h - low_4h
    tr2 = (high_4h - close_4h.shift(1)).abs()
    tr3 = (low_4h - close_4h.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr.ewm(alpha=1/20, min_periods=20, adjust=False).mean()
    atr_filter_val = atr_4h / 1.5
    atr_filt_4h = (low_4h - high_4h.shift(2) > atr_filter_val) | (low_4h.shift(2) - high_4h > atr_filter_val)

    # Trend filter: ta.sma(close_4h, 54) > ta.sma(close_4h[1], 54)
    sma_54 = close_4h.rolling(54, min_periods=54).mean()
    loc2_4h = sma_54 > sma_54.shift(1)

    # FVG conditions (default inp1=inp2=inp3=0 so all filters default to true)
    bfvg_4h = (low_4h > high_4h.shift(2)) & vol_filt_4h & atr_filt_4h & loc2_4h
    sfvg_4h = (high_4h < low_4h.shift(2)) & vol_filt_4h & atr_filt_4h & ~loc2_4h

    # Map 4H conditions back to 15m bars
    bfvg_4h_expanded = bfvg_4h.reindex(df.index, method='ffill').fillna(False)
    sfvg_4h_expanded = sfvg_4h.reindex(df.index, method='ffill').fillna(False)

    # London trading windows (07:45-11:45 and 14:00-14:45 UTC)
    time_minutes = hours * 60 + minutes
    in_window1 = (time_minutes >= 7 * 60 + 45) & (time_minutes < 11 * 60 + 45)
    in_window2 = (time_minutes >= 14 * 60) & (time_minutes < 14 * 60 + 45)
    in_trading_window = in_window1 | in_window2

    # State for sharp turn detection
    last_fvg = 0

    for i in range(len(df)):
        ts = df['time'].iloc[i]
        entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()

        if is_new_4h.iloc[i] and in_trading_window.iloc[i]:
            # Sharp turn logic
            if bfvg_4h_expanded.iloc[i] and last_fvg == -1:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(ts),
                    'entry_time': entry_time,
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                })
                trade_num += 1
                last_fvg = 1
            elif sfvg_4h_expanded.iloc[i] and last_fvg == 1:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(ts),
                    'entry_time': entry_time,
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                })
                trade_num += 1
                last_fvg = -1
            elif bfvg_4h_expanded.iloc[i]:
                last_fvg = 1
            elif sfvg_4h_expanded.iloc[i]:
                last_fvg = -1

    return entries