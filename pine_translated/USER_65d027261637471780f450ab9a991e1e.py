import pandas as pd
import numpy as np
from datetime import datetime, timezone

def ta_atr(high, low, close, length=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    alpha = 1.0 / length
    atr = pd.Series(index=tr.index, dtype=float)
    atr.iloc[length - 1] = tr.iloc[:length].mean()
    for i in range(length, len(tr)):
        atr.iloc[i] = alpha * tr.iloc[i] + (1 - alpha) * atr.iloc[i - 1]
    return atr

def ta_rsi(prices, length=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = pd.Series(index=gain.index, dtype=float)
    avg_loss = pd.Series(index=loss.index, dtype=float)
    avg_gain.iloc[length - 1] = gain.iloc[:length].mean()
    avg_loss.iloc[length - 1] = loss.iloc[:length].mean()
    alpha = 1.0 / length
    for i in range(length, len(gain)):
        avg_gain.iloc[i] = alpha * gain.iloc[i] + (1 - alpha) * avg_gain.iloc[i - 1]
        avg_loss.iloc[i] = alpha * loss.iloc[i] + (1 - alpha) * avg_loss.iloc[i - 1]
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

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

    # Filters (all disabled as per default in Pine Script)
    volfilt11 = True
    atrfilt11 = True
    locfiltb11 = True
    locfilts11 = True

    # Fair Value Gap conditions
    bfvg11 = df['low'] > df['high'].shift(2)
    sfvg11 = df['high'] < df['low'].shift(2)

    # Filter with ATR
    if atrfilt11:
        atr211 = ta_atr(df['high'], df['low'], df['close'], 20) / 1.5
        bull_atr_cond = (df['low'] - df['high'].shift(2)) > atr211
        bear_atr_cond = (df['low'].shift(2) - df['high']) > atr211
        bfvg11 = bfvg11 & (bull_atr_cond | bear_atr_cond)
        sfvg11 = sfvg11 & (bull_atr_cond | bear_atr_cond)

    # Filter with trend
    loc11 = df['close'].ewm(span=54, adjust=False).mean()
    loc211 = loc11 > loc11.shift(1)
    locfiltb11 = loc211 if atrfilt11 else True
    locfilts11 = ~loc211 if atrfilt11 else True
    bfvg11 = bfvg11 & locfiltb11
    sfvg11 = sfvg11 & locfilts11

    # Filter with volume
    if volfilt11:
        vol_sma = df['volume'].rolling(9).mean() * 1.5
        bfvg11 = bfvg11 & (df['volume'].shift(1) > vol_sma)
        sfvg11 = sfvg11 & (df['volume'].shift(1) > vol_sma)

    # Swing detection using daily data
    dailyHigh21 = df['high'].shift(1)
    dailyLow21 = df['low'].shift(1)
    dailyHigh22 = df['high'].shift(2)
    dailyLow22 = df['low'].shift(2)

    is_swing_high11 = (dailyHigh21 < dailyHigh22) & (df['high'].shift(3) < dailyHigh22) & (df['high'].shift(4) < dailyHigh22)
    is_swing_low11 = (dailyLow21 > dailyLow22) & (df['low'].shift(3) > dailyLow22) & (df['low'].shift(4) > dailyLow22)

    last_swing_high11 = np.where(is_swing_high11, dailyHigh22, np.nan)
    last_swing_low11 = np.where(is_swing_low11, dailyLow22, np.nan)
    last_swing_high11 = pd.Series(last_swing_high11).ffill().bfill()
    last_swing_low11 = pd.Series(last_swing_low11).ffill().bfill()

    lastSwingType11 = np.where(is_swing_high11, "dailyHigh", np.where(is_swing_low11, "dailyLow", np.nan))
    lastSwingType11 = pd.Series(lastSwingType11).ffill().bfill()

    isBullishLeg11 = bfvg11 & (lastSwingType11 == "dailyLow")
    isBearishLeg11 = sfvg11 & (lastSwingType11 == "dailyHigh")

    # Supertrend calculation
    superTrendPeriod = 10
    superTrendMultiplier = 3
    atr_st = ta_atr(df['high'], df['low'], df['close'], superTrendPeriod)
    hl2 = (df['high'] + df['low']) / 2
    upperBand = hl2 + (superTrendMultiplier * atr_st)
    lowerBand = hl2 - (superTrendMultiplier * atr_st)

    superTrendDirection = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upperBand.iloc[i]:
            superTrendDirection.iloc[i] = 1
        elif df['close'].iloc[i] < lowerBand.iloc[i]:
            superTrendDirection.iloc[i] = -1
        else:
            superTrendDirection.iloc[i] = superTrendDirection.iloc[i-1] if not pd.isna(superTrendDirection.iloc[i-1]) else 0

    isSuperTrendBullish = superTrendDirection == 1
    isSuperTrendBearish = superTrendDirection == -1

    # Time window (Europe/London timezone)
    timestamps = pd.to_datetime(df['time'], unit='s', utc=True)
    hour = timestamps.dt.hour
    minute = timestamps.dt.minute

    morning_start = (hour == 6) & (minute >= 45)
    morning_end = (hour < 9) | ((hour == 9) & (minute < 45))
    isWithinMorningWindow = morning_start & morning_end

    afternoon_start = (hour == 14) & (minute >= 45)
    afternoon_end = (hour < 16) | ((hour == 16) & (minute < 45))
    isWithinAfternoonWindow = afternoon_start & afternoon_end

    in_trading_window = isWithinMorningWindow | isWithinAfternoonWindow

    # Previous day high/low detection
    newDay = df['time'].diff() >= 86400

    pdHigh = pd.Series(np.nan, index=df.index)
    pdLow = pd.Series(np.nan, index=df.index)

    if len(df) > 0:
        tempHigh = df['high'].cummax().where(newDay.cumsum() > 0, np.nan)
        tempLow = df['low'].cummin().where(newDay.cumsum() > 0, np.nan)

        pdHigh_vals = np.full(len(df), np.nan)
        pdLow_vals = np.full(len(df), np.nan)
        prev_high = np.nan
        prev_low = np.nan

        for i in range(1, len(df)):
            if newDay.iloc[i]:
                pdHigh_vals[i] = prev_high
                pdLow_vals[i] = prev_low
            prev_high = df['high'].iloc[i] if pd.isna(tempHigh.iloc[i]) else max(prev_high, df['high'].iloc[i]) if not pd.isna(prev_high) else df['high'].iloc[i]
            prev_low = df['low'].iloc[i] if pd.isna(tempLow.iloc[i]) else min(prev_low, df['low'].iloc[i]) if not pd.isna(prev_low) else df['low'].iloc[i]

        pdHigh = pd.Series(pdHigh_vals, index=df.index)
        pdLow = pd.Series(pdLow_vals, index=df.index)

    # Entry conditions
    long_conditions = bfvg11 & isBullishLeg11 & isSuperTrendBullish & in_trading_window
    short_conditions = sfvg11 & isBearishLeg11 & isSuperTrendBearish & in_trading_window

    # Convert to numpy for efficient iteration
    long_conditions_arr = long_conditions.to_numpy()
    short_conditions_arr = short_conditions.to_numpy()

    # Sweep state (once per day reset handled by newDay)
    sweptHigh_arr = np.zeros(len(df), dtype=bool)
    sweptLow_arr = np.zeros(len(df), dtype=bool)
    newDay_arr = newDay.to_numpy()
    pdHigh_arr = pdHigh.to_numpy()
    pdLow_arr = pdLow.to_numpy()

    entries = []
    trade_num = 1

    for i in range(20, len(df)):
        # Reset sweep flags on new day
        if newDay_arr[i]:
            sweptHigh_arr[i] = False
            sweptLow_arr[i] = False
        else:
            sweptHigh_arr[i] = sweptHigh_arr[i-1]
            sweptLow_arr[i] = sweptLow_arr[i-1]

        # Check sweeps (for potential filtering - not required for entries)
        if not pd.isna(pdHigh_arr[i]) and not pd.isna(pdHigh_arr[i]):
            if not sweptHigh_arr[i] and df['high'].iloc[i] > pdHigh_arr[i]:
                sweptHigh_arr[i] = True
            if not sweptLow_arr[i] and df['low'].iloc[i] < pdLow_arr[i]:
                sweptLow_arr[i] = True

        if long_conditions_arr[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_conditions_arr[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries