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
    # Parameters (matching Pine Script inputs)
    bb = 20  # Lookback Range
    input_retSince = 2  # Bars Since Breakout
    input_retValid = 2  # Retest Detection Limiter

    # Helper: Wilder RSI
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Helper: Wilder ATR
    def wilder_atr(df, length):
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        return atr

    # CMO calculation
    def calc_cmo(src, length):
        mom = src.diff()
        pos_sum = mom.where(mom > 0, 0.0).rolling(window=length).sum()
        neg_sum = (-mom.where(mom < 0, 0.0)).rolling(window=length).sum()
        cmo = 100 * (pos_sum - neg_sum) / (pos_sum + neg_sum)
        return cmo

    # Calculate pivot points
    def calc_pivotlow(low_series, bb):
        pl = low_series.rolling(window=2*bb+1).min()
        pl = pl.where(low_series == low_series.rolling(window=2*bb+1).min(), np.nan)
        return pl

    def calc_pivothigh(high_series, bb):
        ph = high_series.rolling(window=2*bb+1).max()
        ph = ph.where(high_series == high_series.rolling(window=2*bb+1).max(), np.nan)
        return ph

    # Calculate boxes (simplified support/resistance boxes)
    sBox_bot = np.full(len(df), np.nan)
    sBox_top = np.full(len(df), np.nan)
    rBox_bot = np.full(len(df), np.nan)
    rBox_top = np.full(len(df), np.nan)

    pl = calc_pivotlow(df['low'], bb)
    ph = calc_pivothigh(df['high'], bb)

    # Box height calculation
    s_yLoc = np.where(df['low'].shift(bb + 1) > df['low'].shift(bb - 1), df['low'].shift(bb - 1), df['low'].shift(bb + 1))
    r_yLoc = np.where(df['high'].shift(bb + 1) > df['high'].shift(bb - 1), df['high'].shift(bb + 1), df['high'].shift(bb - 1))

    # Calculate breakout conditions
    # cu = crossunder(close, sBox_bot) - breakout below support
    # co = crossover(close, rBox_top) - breakout above resistance
    cu = np.full(len(df), False)
    co = np.full(len(df), False)

    for i in range(bb + 1, len(df)):
        prev_pl_idx = i - bb
        if prev_pl_idx >= 0 and not pd.isna(pl.iloc[prev_pl_idx]):
            sBot_val = df['low'].iloc[prev_pl_idx]
            sTop_val = s_yLoc[prev_pl_idx]
            if not pd.isna(sTop_val):
                sBox_bot[i] = sBot_val
                sBox_top[i] = sTop_val
                if i > 0:
                    cu[i] = df['close'].iloc[i] < sBot_val and df['close'].iloc[i-1] >= sBot_val

        prev_ph_idx = i - bb
        if prev_ph_idx >= 0 and not pd.isna(ph.iloc[prev_ph_idx]):
            rTop_val = df['high'].iloc[prev_ph_idx]
            rBot_val = r_yLoc[prev_ph_idx]
            if not pd.isna(rBot_val):
                rBox_top[i] = rTop_val
                rBox_bot[i] = rBot_val
                if i > 0:
                    co[i] = df['close'].iloc[i] > rTop_val and df['close'].iloc[i-1] <= rTop_val

    # Convert to pandas Series for easier handling
    cu_series = pd.Series(cu, index=df.index)
    co_series = pd.Series(co, index=df.index)

    # Calculate retest conditions
    # retestCondition(breakout, condition) => barssince(na(breakout)) > input_retSince and condition
    def barssince(cond_series):
        result = np.full(len(df), np.nan)
        count = 0
        found = False
        for i in range(len(df)):
            if cond_series.iloc[i]:
                count = 0
                found = True
            elif found:
                count += 1
                result[i] = count
        return pd.Series(result, index=df.index)

    sBreak = np.full(len(df), False)
    rBreak = np.full(len(df), False)

    for i in range(len(df)):
        if cu_series.iloc[i] and pd.isna(sBreak[i-1] if i > 0 else False):
            sBreak[i] = True
        elif pd.notna(pl.iloc[i]) if i >= bb else False:
            sBreak[i] = False

        if co_series.iloc[i] and pd.isna(rBreak[i-1] if i > 0 else False):
            rBreak[i] = True
        elif pd.notna(ph.iloc[i]) if i >= bb else False:
            rBreak[i] = False

    sBreak_series = pd.Series(sBreak, index=df.index)
    rBreak_series = pd.Series(rBreak, index=df.index)

    # Retest conditions for support (s1-s4)
    s1 = barssince(sBreak_series) > input_retSince & (df['high'] >= sBox_top) & (df['close'] <= sBox_bot)
    s2 = barssince(sBreak_series) > input_retSince & (df['high'] >= sBox_top) & (df['close'] >= sBox_bot) & (df['close'] <= sBox_top)
    s3 = barssince(sBreak_series) > input_retSince & (df['high'] >= sBox_bot) & (df['high'] <= sBox_top)
    s4 = barssince(sBreak_series) > input_retSince & (df['high'] >= sBox_bot) & (df['high'] <= sBox_top) & (df['close'] < sBox_bot)

    # Retest conditions for resistance (r1-r4)
    r1 = barssince(rBreak_series) > input_retSince & (df['low'] <= rBox_bot) & (df['close'] >= rBox_top)
    r2 = barssince(rBreak_series) > input_retSince & (df['low'] <= rBox_bot) & (df['close'] <= rBox_top) & (df['close'] >= rBox_bot)
    r3 = barssince(rBreak_series) > input_retSince & (df['low'] <= rBox_top) & (df['low'] >= rBox_bot)
    r4 = barssince(rBreak_series) > input_retSince & (df['low'] <= rBox_top) & (df['low'] >= rBox_bot) & (df['close'] > rBox_top)

    # Combined signals
    long_signal = co_series | r1 | r2 | r3 | r4  # Long on resistance breakout or resistance retest
    short_signal = cu_series | s1 | s2 | s3 | s4  # Short on support breakout or support retest

    # Build entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if i < bb:
            continue

        direction = None
        if long_signal.iloc[i]:
            direction = 'long'
        elif short_signal.iloc[i]:
            direction = 'short'

        if direction:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]

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

    return entries