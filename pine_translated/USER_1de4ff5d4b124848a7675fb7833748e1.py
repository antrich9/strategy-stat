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
    atrLength = 14
    atrMultiplier = 1.5
    takeProfitRatio = 1.5
    tradeDirection = 'Both'
    fastLength = 12
    slowLength = 26
    signalSmoothing = 9
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    input_breakout = True
    input_retest = True

    close = df['close']
    high = df['high']
    low = df['low']

    # Zero-Lag EMA
    def zlema(src, length):
        lag = int(length / 2)
        ema2 = src.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        return 2 * ema2 - ema3

    fastEMA = zlema(close, fastLength)
    slowEMA = zlema(close, slowLength)
    macd = fastEMA - slowEMA
    signal = macd.ewm(span=signalSmoothing, adjust=False).mean()
    hist = macd - signal

    # Wilder ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/atrLength, adjust=False).mean()

    bb = input_lookback

    # Pivot points
    pl = low.rolling(window=bb+1, min_periods=bb+1).min().shift(bb)
    ph = high.rolling(window=bb+1, min_periods=bb+1).max().shift(bb)

    # Box heights
    s_yLoc = pd.where(low.shift(bb+1) > low.shift(bb-1), low.shift(bb-1), low.shift(bb+1))
    r_yLoc = pd.where(high.shift(bb+1) > high.shift(bb-1), high.shift(bb+1), high.shift(bb-1))

    # Support/Resistance boxes (simplified - using rolling box)
    sTop = pd.Series(np.where(low.shift(bb+1) > low.shift(bb-1), low.shift(bb-1), low.shift(bb+1)), index=df.index)
    sBot = pl
    rTop = ph
    rBot = pd.Series(np.where(high.shift(bb+1) > high.shift(bb-1), high.shift(bb+1), high.shift(bb-1)), index=df.index)

    # Breakout detection
    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)

    for i in range(bb + 2, len(df)):
        sBot_val = sBot.iloc[i] if not pd.isna(sBot.iloc[i]) else None
        rTop_val = rTop.iloc[i] if not pd.isna(rTop.iloc[i]) else None

        if sBot_val is not None and close.iloc[i] < sBot_val:
            sBreak.iloc[i] = True

        if rTop_val is not None and close.iloc[i] > rTop_val:
            rBreak.iloc[i] = True

    # Retest conditions for support (s1, s2, s3, s4)
    def retestCondition(breakout_series, condition_mask):
        bars_since_breakout = pd.Series(np.nan, index=df.index)
        result = pd.Series(False, index=df.index)

        for i in range(1, len(df)):
            if breakout_series.iloc[i]:
                bars_since_breakout.iloc[i] = 0
            elif not pd.isna(bars_since_breakout.iloc[i-1]):
                bars_since_breakout.iloc[i] = bars_since_breakout.iloc[i-1] + 1

            if not pd.isna(bars_since_breakout.iloc[i]) and bars_since_breakout.iloc[i] > input_retSince:
                result.iloc[i] = condition_mask.iloc[i] if i < len(condition_mask) else False

        return result

    s1 = pd.Series(False, index=df.index)
    s2 = pd.Series(False, index=df.index)
    s3 = pd.Series(False, index=df.index)
    s4 = pd.Series(False, index=df.index)

    r1 = pd.Series(False, index=df.index)
    r2 = pd.Series(False, index=df.index)
    r3 = pd.Series(False, index=df.index)
    r4 = pd.Series(False, index=df.index)

    for i in range(bb + 2, len(df)):
        if sBreak.iloc[i] and not pd.isna(sTop.iloc[i]) and not pd.isna(sBot.iloc[i]):
            sTop_val = sTop.iloc[i]
            sBot_val = sBot.iloc[i]

            if high.iloc[i] >= sTop_val and close.iloc[i] <= sBot_val:
                s1.iloc[i] = True
            if high.iloc[i] >= sTop_val and close.iloc[i] >= sBot_val and close.iloc[i] <= sTop_val:
                s2.iloc[i] = True
            if high.iloc[i] >= sBot_val and high.iloc[i] <= sTop_val:
                s3.iloc[i] = True
            if high.iloc[i] >= sBot_val and high.iloc[i] <= sTop_val and close.iloc[i] < sBot_val:
                s4.iloc[i] = True

        if rBreak.iloc[i] and not pd.isna(rTop.iloc[i]) and not pd.isna(rBot.iloc[i]):
            rTop_val = rTop.iloc[i]
            rBot_val = rBot.iloc[i]

            if low.iloc[i] <= rBot_val and close.iloc[i] >= rTop_val:
                r1.iloc[i] = True
            if low.iloc[i] <= rBot_val and close.iloc[i] <= rTop_val and close.iloc[i] >= rBot_val:
                r2.iloc[i] = True
            if low.iloc[i] <= rTop_val and low.iloc[i] >= rBot_val:
                r3.iloc[i] = True
            if low.iloc[i] <= rTop_val and low.iloc[i] >= rBot_val and close.iloc[i] > rTop_val:
                r4.iloc[i] = True

    # Retest valid (simplified - detecting when retest occurs after breakout)
    sRetValid = pd.Series(False, index=df.index)
    rRetValid = pd.Series(False, index=df.index)

    for i in range(1, len(df)):
        if sBreak.iloc[i]:
            # Find next retest within valid bars
            for j in range(i+1, min(i+1+input_retValid, len(df))):
                if (s1.iloc[j] or s2.iloc[j] or s3.iloc[j] or s4.iloc[j]):
                    sRetValid.iloc[j] = True
                    break

        if rBreak.iloc[i]:
            for j in range(i+1, min(i+1+input_retValid, len(df))):
                if (r1.iloc[j] or r2.iloc[j] or r3.iloc[j] or r4.iloc[j]):
                    rRetValid.iloc[j] = True
                    break

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(bb + 2, len(df)):
        if pd.isna(atr.iloc[i]):
            continue

        # Long entry on support retest
        if (tradeDirection in ['Long', 'Both']) and sRetValid.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

        # Short entry on resistance retest
        if (tradeDirection in ['Short', 'Both']) and rRetValid.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

    return entries