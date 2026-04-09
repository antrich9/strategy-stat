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
    # Parameters from Pine Script
    PP = 5  # Pivot Period
    atrLength = 55  # ATR length for ZigZag
    atrStopLength = 14  # ATR length for stop loss
    atrMultiplier = 1.5  # ATR multiplier for stop loss distance

    # Helper: Wilder RSI
    def wilders_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Helper: Wilder ATR
    def wilders_atr(df, period):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr

    # Calculate ATR
    atr = wilders_atr(df, atrLength)
    atr_stop = wilders_atr(df, atrStopLength)

    # Pivot High and Low using rolling window
    def pivot_high(series, left, right):
        result = pd.Series(np.nan, index=series.index)
        for i in range(right, len(series) - left):
            window = series.iloc[i - right:i + left + 1]
            if window.idxmax() == i:
                result.iloc[i] = series.iloc[i]
        return result

    def pivot_low(series, left, right):
        result = pd.Series(np.nan, index=series.index)
        for i in range(right, len(series) - left):
            window = series.iloc[i - right:i + left + 1]
            if window.idxmin() == i:
                result.iloc[i] = series.iloc[i]
        return result

    high_pivot = pivot_high(df['high'], PP, PP)
    low_pivot = pivot_low(df['low'], PP, PP)

    # Initialize arrays to track ZigZag pivots (simplified)
    pivot_types = [None] * len(df)
    pivot_values = [np.nan] * len(df)
    pivot_indices = [0] * len(df)
    array_size = 0

    # Major and Minor structure tracking
    major_high = pd.Series(np.nan, index=df.index)
    major_low = pd.Series(np.nan, index=df.index)
    minor_high = pd.Series(np.nan, index=df.index)
    minor_low = pd.Series(np.nan, index=df.index)

    # BoS and ChoCh flags
    bullish_major_bos = pd.Series(False, index=df.index)
    bearish_major_bos = pd.Series(False, index=df.index)
    bullish_major_choch = pd.Series(False, index=df.index)
    bearish_major_choch = pd.Series(False, index=df.index)

    # Track last major/minor pivot values
    last_major_high_idx = 0
    last_major_low_idx = 0
    last_minor_high_idx = 0
    last_minor_low_idx = 0

    # Simplified ZigZag and structure detection
    for i in range(PP * 2 + 1, len(df)):
        # Update pivots
        if not np.isnan(high_pivot.iloc[i]):
            pivot_types[i] = 'H'
            pivot_values[i] = df['high'].iloc[i]
            pivot_indices[i] = i

            # Update major/minor high
            if array_size == 0 or pivot_values[i] > pivot_values[last_major_high_idx]:
                if array_size > 0:
                    last_major_high_idx = i
            else:
                last_minor_high_idx = i

        if not np.isnan(low_pivot.iloc[i]):
            pivot_types[i] = 'L'
            pivot_values[i] = df['low'].iloc[i]
            pivot_indices[i] = i

            # Update major/minor low
            if array_size == 0 or pivot_values[i] < pivot_values[last_major_low_idx]:
                if array_size > 0:
                    last_major_low_idx = i
            else:
                last_minor_low_idx = i

        array_size = i + 1

        # Detect Major BoS (Break of Structure)
        if last_major_high_idx > 0 and i - last_major_high_idx >= PP:
            prev_major_high = major_high.shift(1).iloc[i]
            if not np.isnan(prev_major_high) and df['close'].iloc[i] > prev_major_high:
                bullish_major_bos.iloc[i] = True

        if last_major_low_idx > 0 and i - last_major_low_idx >= PP:
            prev_major_low = major_low.shift(1).iloc[i]
            if not np.isnan(prev_major_low) and df['close'].iloc[i] < prev_major_low:
                bearish_major_bos.iloc[i] = True

        # Update major levels
        if pivot_types[i] == 'H':
            major_high.iloc[i] = pivot_values[i]
        else:
            major_high.iloc[i] = major_high.iloc[i-1] if not np.isnan(major_high.iloc[i-1]) else np.nan

        if pivot_types[i] == 'L':
            major_low.iloc[i] = pivot_values[i]
        else:
            major_low.iloc[i] = major_low.iloc[i-1] if not np.isnan(major_low.iloc[i-1]) else np.nan

    # Calculate EMA for trend
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    ema200 = df['close'].ewm(span=200, adjust=False).mean()

    # Trend: Bullish when EMA50 > EMA200
    uptrend = ema50 > ema200
    downtrend = ema50 < ema200

    # Higher high / Higher low detection for trend
    hh = pd.Series(np.nan, index=df.index)
    ll = pd.Series(np.nan, index=df.index)
    for i in range(PP * 2 + 1, len(df)):
        if not np.isnan(high_pivot.iloc[i]):
            prev_hh = hh.iloc[i-1] if not np.isnan(hh.iloc[i-1]) else 0
            if df['high'].iloc[i] > prev_hh:
                hh.iloc[i] = df['high'].iloc[i]
            else:
                hh.iloc[i] = prev_hh
        else:
            hh.iloc[i] = hh.iloc[i-1] if not np.isnan(hh.iloc[i-1]) else np.nan

        if not np.isnan(low_pivot.iloc[i]):
            prev_ll = ll.iloc[i-1] if not np.isnan(ll.iloc[i-1]) else float('inf')
            if df['low'].iloc[i] < prev_ll:
                ll.iloc[i] = df['low'].iloc[i]
            else:
                ll.iloc[i] = prev_ll
        else:
            ll.iloc[i] = ll.iloc[i-1] if not np.isnan(ll.iloc[i-1]) else np.nan

    # Entry conditions based on strategy name and logic
    # Long: Major Bullish BoS or ChoCh with higher high/lower low confirmation
    long_condition = (bullish_major_bos) | (bullish_major_choch)

    # Short: Major Bearish BoS or ChoCh with lower low/lower high confirmation
    short_condition = (bearish_major_bos) | (bearish_major_choch)

    # Additional filter: require trend alignment (optional based on strategy)
    # In uptrend, prefer longs; in downtrend, prefer shorts
    # long_condition = long_condition & uptrend
    # short_condition = short_condition & downtrend

    # Build result list
    entries = []
    trade_num = 1

    for i in range(PP * 2 + 1, len(df)):
        # Skip if ATR is NaN
        if np.isnan(atr.iloc[i]) or np.isnan(atr_stop.iloc[i]):
            continue

        entry_price = df['close'].iloc[i]
        ts = int(df['time'].iloc[i])

        if long_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000 if ts > 1e10 else ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

        if short_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000 if ts > 1e10 else ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

    return entries