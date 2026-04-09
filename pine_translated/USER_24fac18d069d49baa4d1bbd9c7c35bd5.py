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

    prd = 2

    def wilder_smooth(series, length):
        alpha = 1.0 / length
        return series.ewm(alpha=alpha, adjust=False).mean()

    def calculate_rsi(src, length):
        delta = src.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = wilder_smooth(gain, length)
        avg_loss = wilder_smooth(loss, length)
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(high, low, close, length):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return wilder_smooth(tr, length)

    def calculate_zigzag(high, low, prd):
        swing_highs = []
        swing_lows = []
        last_high_idx = -1
        last_low_idx = -1
        dir = 0

        for i in range(prd, len(high)):
            lookback = high.iloc[i-prd:i+1]
            if lookback.idxmax() == i - prd:
                h_idx = i - prd
                h_val = high.iloc[h_idx]
                if dir == -1 or dir == 0:
                    swing_highs.append(h_idx)
                    dir = 1
                elif dir == 1 and h_val > high.iloc[last_high_idx]:
                    swing_highs[-1] = h_idx
                last_high_idx = len(swing_highs) - 1

            lookback_low = low.iloc[i-prd:i+1]
            if lookback_low.idxmin() == i - prd:
                l_idx = i - prd
                l_val = low.iloc[l_idx]
                if dir == 1 or dir == 0:
                    swing_lows.append(l_idx)
                    dir = -1
                elif dir == -1 and l_val < low.iloc[last_low_idx]:
                    swing_lows[-1] = l_idx
                last_low_idx = len(swing_lows) - 1

        return swing_highs, swing_lows

    swing_highs, swing_lows = calculate_zigzag(df['high'], df['low'], prd)

    confirmed_swing_high = pd.Series(np.nan, index=df.index)
    confirmed_swing_low = pd.Series(np.nan, index=df.index)
    fib_50 = pd.Series(np.nan, index=df.index)

    for i in range(prd, len(df)):
        if df['high'].iloc[i-prd:i+1].max() == df['high'].iloc[i-prd]:
            confirmed_swing_high.iloc[i] = df['high'].iloc[i-prd]
        if df['low'].iloc[i-prd:i+1].min() == df['low'].iloc[i-prd]:
            confirmed_swing_low.iloc[i] = df['low'].iloc[i-prd]

    if len(swing_highs) >= 1 and len(swing_lows) >= 1:
        fib_0 = df['low'].iloc[swing_lows[-1]]
        fib_1 = df['high'].iloc[swing_highs[-1]]
        fib_val = fib_0 + (fib_1 - fib_0) * 0.5
        fib_50.iloc[swing_highs[-1]:] = fib_val

    atr14 = calculate_atr(df['high'], df['low'], df['close'], 14)
    atr20 = calculate_atr(df['high'], df['low'], df['close'], 20)
    vol_sma9 = df['volume'].rolling(9).mean()
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)

    vol_filt = df['volume'] > vol_sma9 * 1.5
    atr2 = atr20 / 1.5
    atr_filt = (df['low'] - df['high'].shift(2) > atr2) | (df['low'].shift(2) - df['high'] > atr2)
    loc_filt_bull = loc2
    loc_filt_bear = ~loc2

    bull_fvg = (df['low'] > df['high'].shift(2)) & vol_filt & atr_filt & loc_filt_bull
    bear_fvg = (df['high'] < df['low'].shift(2)) & vol_filt & atr_filt & loc_filt_bear

    bullsharp = bull_fvg.copy()
    bearsharp = bear_fvg.copy()

    bull_fvg_confirmed = (df['high'] < df['low'].shift(2)) & (df['close'].shift(1) < df['low'].shift(2))
    bear_fvg_confirmed = (df['low'] > df['high'].shift(2)) & (df['close'].shift(1) > df['high'].shift(2))

    lookback_bars = 12
    bull_since = pd.Series(np.nan, index=df.index)
    bear_since = pd.Series(np.nan, index=df.index)

    bull_since_valid = False
    for i in range(len(df)):
        if bull_fvg_confirmed.iloc[i]:
            bull_since_valid = True
            bull_since.iloc[i] = 0
        elif bull_since_valid:
            prev_val = bull_since.iloc[i-1] if not pd.isna(bull_since.iloc[i-1]) else 0
            bull_since.iloc[i] = prev_val + 1

    bear_since_valid = False
    for i in range(len(df)):
        if bear_fvg_confirmed.iloc[i]:
            bear_since_valid = True
            bear_since.iloc[i] = 0
        elif bear_since_valid:
            prev_val = bear_since.iloc[i-1] if not pd.isna(bear_since.iloc[i-1]) else 0
            bear_since.iloc[i] = prev_val + 1

    bull_result = pd.Series(False, index=df.index)
    bear_result = pd.Series(False, index=df.index)

    for i in range(lookback_bars + 2, len(df)):
        bs = bull_since.iloc[i]
        if not pd.isna(bs) and bs <= lookback_bars and bull_fvg.iloc[i]:
            combined_low_val = max(df['high'].iloc[i-int(bs)], df['high'].iloc[i-2])
            combined_high_val = min(df['low'].iloc[i-int(bs)+2], df['low'].iloc[i])
            if combined_high_val - combined_low_val >= 0:
                bull_result.iloc[i] = True

        bes = bear_since.iloc[i]
        if not pd.isna(bes) and bes <= lookback_bars and bear_fvg.iloc[i]:
            combined_low_val = max(df['high'].iloc[i], df['high'].iloc[i-int(bes)+2])
            combined_high_val = min(df['low'].iloc[i-int(bes)], df['low'].iloc[i-2])
            if combined_high_val - combined_low_val >= 0:
                bear_result.iloc[i] = True

    bull_entry_cond = bullsharp & confirmed_swing_high.notna() & (df['close'] > confirmed_swing_high) & (df['close'] < fib_50)
    bear_entry_cond = bearsharp & confirmed_swing_low.notna() & (df['close'] < confirmed_swing_low) & (df['close'] > fib_50)

    bull_entry_cond2 = bull_result & (df['close'] < fib_50)
    bear_entry_cond2 = bear_result & (df['close'] > fib_50)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if (bull_entry_cond.iloc[i] or bull_entry_cond2.iloc[i]) and not pd.isna(fib_50.iloc[i]):
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    for i in range(len(df)):
        if (bear_entry_cond.iloc[i] or bear_entry_cond2.iloc[i]) and not pd.isna(fib_50.iloc[i]):
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries