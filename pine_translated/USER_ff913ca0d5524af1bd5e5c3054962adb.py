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
    # Wilder RSI implementation
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Wilder ATR implementation
    def wilder_atr(high, low, close, length):
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
        return atr

    # Bollinger Bands
    def bollinger_bands(close, length, mult):
        sma = close.rolling(length).mean()
        std = close.rolling(length).std()
        upper = sma + mult * std
        lower = sma - mult * std
        return sma, upper, lower

    # Pivot high/low detection
    def pivot_high(series, left_len, right_len):
        pivot = series.rolling(right_len, center=True).max().shift(1).rolling(left_len).max()
        return series[(series == pivot) & (series.shift(right_len) == pivot)].index

    def pivot_low(series, left_len, right_len):
        pivot = series.rolling(right_len, center=True).min().shift(1).rolling(left_len).min()
        return series[(series == pivot) & (series.shift(right_len) == pivot)].index

    # Calculate indicators
    bb_sma, bb_upper, bb_lower = bollinger_bands(df['close'], 20, 2.0)
    atr_55 = wilder_atr(df['high'], df['low'], df['close'], 55)
    atr_10 = wilder_atr(df['high'], df['low'], df['close'], 10)

    # State variables
    PP = 5  # Pivot Period

    # Calculate pivots
    pivot_high_bar = pd.Series(index=df.index, dtype=float)
    pivot_low_bar = pd.Series(index=df.index, dtype=float)

    for i in range(PP, len(df) - PP):
        window_high = df['high'].iloc[i-PP:i+1].max()
        window_low = df['low'].iloc[i-PP:i+1].min()
        if df['high'].iloc[i] == window_high:
            pivot_high_bar.iloc[i] = df['high'].iloc[i]
        if df['low'].iloc[i] == window_low:
            pivot_low_bar.iloc[i] = df['low'].iloc[i]

    # Market structure state tracking
    # State constants
    BBplus = 0
    signUP = 1
    cnclUP = 2
    LL1break = 3
    LL2break = 4
    SW1breakUP = 5
    SW2breakUP = 6
    tpUP1 = 7
    tpUP2 = 8
    tpUP3 = 9
    BB_endBl = 10
    BB_min = 11
    signDN = 12
    cnclDN = 13
    HL1break = 14
    HL2break = 15
    SW1breakDN = 16
    SW2breakDN = 17
    tpDN1 = 18
    tpDN2 = 19
    tpDN3 = 20
    BB_endBr = 21

    # Initialize state tracking
    state = pd.Series(index=df.index, dtype=int)
    state.iloc[:] = -1

    Bullish_BOS = pd.Series(False, index=df.index)
    Bearish_BOS = pd.Series(False, index=df.index)
    Bullish_ChoCh = pd.Series(False, index=df.index)
    Bearish_ChoCh = pd.Series(False, index=df.index)

    # Structure tracking variables
    var_last_high_idx = 0
    var_last_low_idx = 0
    var_last_high_val = df['high'].iloc[0]
    var_last_low_val = df['low'].iloc[0]

    # Track swing highs and lows
    swing_highs = pd.Series(index=df.index, dtype=float)
    swing_lows = pd.Series(index=df.index, dtype=float)
    swing_highs.iloc[:] = np.nan
    swing_lows.iloc[:] = np.nan

    # Detect swing points
    for i in range(5, len(df) - 5):
        if df['high'].iloc[i] == df['high'].iloc[i-5:i+6].max():
            swing_highs.iloc[i] = df['high'].iloc[i]
        if df['low'].iloc[i] == df['low'].iloc[i-5:i+6].min():
            swing_lows.iloc[i] = df['low'].iloc[i]

    # Structure tracking
    last_swing_high = df['high'].iloc[0]
    last_swing_low = df['low'].iloc[0]
    last_swing_high_idx = 0
    last_swing_low_idx = 0

    # MSS tracking
    Minor_High = pd.Series(index=df.index, dtype=float)
    Minor_Low = pd.Series(index=df.index, dtype=float)
    Major_High = pd.Series(index=df.index, dtype=float)
    Major_Low = pd.Series(index=df.index, dtype=float)

    for i in range(PP, len(df)):
        # Find last swing high/low
        for j in range(i - 1, PP - 1, -1):
            if not pd.isna(swing_highs.iloc[j]):
                last_swing_high = swing_highs.iloc[j]
                last_swing_high_idx = j
                break
        for j in range(i - 1, PP - 1, -1):
            if not pd.isna(swing_lows.iloc[j]):
                last_swing_low = swing_lows.iloc[j]
                last_swing_low_idx = j
                break

        # Store current structure levels
        Minor_High.iloc[i] = last_swing_high
        Minor_Low.iloc[i] = last_swing_low

    # BB condition detection
    bullishBB = df['close'] > bb_upper
    bearishBB = df['close'] < bb_lower

    # Structure break conditions
    # Lower low break (bullish structure break)
    ll_break_up = pd.Series(False, index=df.index)
    # Higher high break (bearish structure break)
    hh_break_dn = pd.Series(False, index=df.index)

    for i in range(PP + 1, len(df)):
        # Check for swing low breaks (for bullish bias)
        for j in range(i - 1, PP, -1):
            if not pd.isna(swing_lows.iloc[j]):
                prev_low = swing_lows.iloc[j]
                for k in range(j - 1, PP, -1):
                    if not pd.isna(swing_lows.iloc[k]):
                        curr_low = swing_lows.iloc[k]
                        if curr_low < prev_low and df['close'].iloc[i] > prev_low:
                            ll_break_up.iloc[i] = True
                        break
                break

        # Check for swing high breaks (for bearish bias)
        for j in range(i - 1, PP, -1):
            if not pd.isna(swing_highs.iloc[j]):
                prev_high = swing_highs.iloc[j]
                for k in range(j - 1, PP, -1):
                    if not pd.isna(swing_highs.iloc[k]):
                        curr_high = swing_highs.iloc[k]
                        if curr_high > prev_high and df['close'].iloc[i] < prev_high:
                            hh_break_dn.iloc[i] = True
                        break
                break

    # Entry signals
    long_entry = pd.Series(False, index=df.index)
    short_entry = pd.Series(False, index=df.index)

    # Long entry: BB bullish + structure confirmation
    for i in range(PP, len(df)):
        if bullishBB.iloc[i]:
            # Look for bullish structure break or uptrend continuation
            if ll_break_up.iloc[i] or (df['close'].iloc[i] > df['close'].iloc[i-1] and df['close'].iloc[i] > df['close'].iloc[max(0, i-5)]):
                long_entry.iloc[i] = True

    # Short entry: BB bearish + structure confirmation
    for i in range(PP, len(df)):
        if bearishBB.iloc[i]:
            # Look for bearish structure break or downtrend continuation
            if hh_break_dn.iloc[i] or (df['close'].iloc[i] < df['close'].iloc[i-1] and df['close'].iloc[i] < df['close'].iloc[max(0, i-5)]):
                short_entry.iloc[i] = True

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(PP, len(df)):
        if long_entry.iloc[i] and not pd.isna(df['close'].iloc[i]):
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        if short_entry.iloc[i] and not pd.isna(df['close'].iloc[i]):
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries