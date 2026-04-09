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
    # Settings (matching Pine Script defaults)
    zzLen = 5
    fibTolerance = 35.0
    minWaveRatio = 0.382
    maxWaveRatio = 0.886
    minWaveSize = 1.0
    signalCooldown = 20
    requireDirectionChange = True
    confirmationBars = 3
    ewoFastLen = 5
    ewoSlowLen = 34
    ewoSignalLen = 5
    ewoThreshold = 13.0
    tradeDir = "Both"
    minConfScore = 50.0
    obLookback = 20
    fvgMinSize = 0.1
    useSMC = True
    useEWO = True
    requireEWOConfirm = False

    n = len(df)
    if n < 50:
        return []

    # EWO Calculation
    ewoFast = df['close'].ewm(span=ewoFastLen, adjust=False).mean()
    ewoSlow = df['close'].ewm(span=ewoSlowLen, adjust=False).mean()
    ewoCalc = (ewoFast / ewoSlow - 1) * 100
    ewoSignalCalc = ewoCalc.ewm(span=ewoSignalLen, adjust=False).mean()

    # EWO Signals
    ewoLongSig = (ewoCalc > ewoSignalCalc) & (ewoCalc.shift(1) <= ewoSignalCalc.shift(1))
    ewoShortSig = (ewoCalc < ewoSignalCalc) & (ewoCalc.shift(1) >= ewoSignalCalc.shift(1))
    ewoStrongL = ewoLongSig & (ewoCalc < -ewoThreshold)
    ewoStrongS = ewoShortSig & (ewoCalc > ewoThreshold)
    ewoBull = ewoCalc > ewoSignalCalc
    ewoBear = ewoCalc < ewoSignalCalc

    # ZigZag implementation
    zz = np.zeros(n)
    zz_dir = np.zeros(n)

    for i in range(zzLen, n):
        high_lookback = df['high'].iloc[i-zzLen:i+1].values
        low_lookback = df['low'].iloc[i-zzLen:i+1].values

        if len(high_lookback) < 2 or len(low_lookback) < 2:
            continue

        max_idx = np.argmax(high_lookback)
        min_idx = np.argmin(low_lookback)

        if max_idx == len(high_lookback) - 1:
            zz[i] = df['high'].iloc[i]
            zz_dir[i] = -1
        elif min_idx == len(low_lookback) - 1:
            zz[i] = df['low'].iloc[i]
            zz_dir[i] = 1

    zz_series = pd.Series(zz, index=df.index)
    zz_dir_series = pd.Series(zz_dir, index=df.index)

    # Global state variables (simulated)
    entries = []
    trade_num = 0
    last_signal_bar = -signalCooldown
    last_direction = 0
    is_active = False
    wave_dir = 0
    w0, w1, w2 = np.nan, np.nan, np.nan
    b0, b1, b2 = 0, 0, 0
    pattern_start_bar = 0
    current_wave = 0

    # EMA50 for confluence
    ema50 = df['close'].ewm(span=50, adjust=False).mean()

    # Volume SMA
    vol_sma20 = df['volume'].rolling(20).mean()

    # Fibonacci validation function
    def is_fib_valid(ratio):
        if ratio >= minWaveRatio and ratio <= maxWaveRatio:
            return True
        tol = fibTolerance / 100
        fib_levels = [0.382, 0.500, 0.618, 0.764]
        for level in fib_levels:
            if level * (1 - tol) <= ratio <= level * (1 + tol):
                return True
        return False

    # Order Block detection (simplified)
    ob_high = pd.Series(np.nan, index=df.index)
    ob_low = pd.Series(np.nan, index=df.index)

    for i in range(obLookback + 2, n):
        for look in range(1, obLookback + 1):
            idx = i - look
            if idx >= 0 and idx < i:
                if df['close'].iloc[idx] < df['open'].iloc[idx] and \
                   df['high'].iloc[idx - 1] > df['high'].iloc[idx] and idx > 0:
                    ob_high.iloc[i] = df['high'].iloc[idx]
                    ob_low.iloc[i] = df['low'].iloc[idx]
                    break

    # Fair Value Gap detection (simplified)
    fvg_high = pd.Series(np.nan, index=df.index)
    fvg_low = pd.Series(np.nan, index=df.index)

    for i in range(3, n):
        for look in range(2, 15):
            idx = i - look
            if idx >= 1 and idx + 1 < i:
                gap_bull = df['low'].iloc[idx - 1] - df['high'].iloc[idx + 1]
                if gap_bull > df['close'].iloc[i] * (fvgMinSize / 100):
                    fvg_high.iloc[i] = df['low'].iloc[idx - 1]
                    fvg_low.iloc[i] = df['high'].iloc[idx + 1]
                    break

    # Pattern detection and entry logic
    for i in range(zzLen * 2 + 5, n):
        if pd.isna(zz_series.iloc[i]) or zz_series.iloc[i] == 0:
            continue

        current_zz = zz_series.iloc[i]
        current_dir = int(zz_dir_series.iloc[i])

        # Detect wave pivots
        if current_wave == 0 and current_dir != 0:
            w0 = current_zz
            b0 = i
            current_wave = 1
        elif current_wave == 1 and current_dir == -1:
            w1 = current_zz
            b1 = i
            current_wave = 2
        elif current_wave == 2 and current_dir == 1:
            w2 = current_zz
            b2 = i
            current_wave = 3
        elif current_wave >= 3:
            if current_dir == -1:
                w0 = w1
                w1 = w2
                w2 = current_zz
                b0 = b1
                b1 = b2
                b2 = i
                current_wave = 3
            elif current_dir == 1:
                w0 = w1
                w1 = w2
                w2 = current_zz
                b0 = b1
                b1 = b2
                b2 = i
                current_wave = 3

        # Entry logic when Wave 2 is detected
        if current_wave == 3 and not pd.isna(w0) and not pd.isna(w1) and not pd.isna(w2):
            wave_size = abs(w1 - w0)
            if wave_size < df['close'].iloc[i] * (minWaveSize / 100):
                continue

            if b2 > b1 and b1 > b0:
                # Bullish Wave 2 (retracement down)
                retracement = (w2 - w0) / wave_size if wave_size != 0 else 0

                if is_fib_valid(retracement):
                    # Calculate confluence score
                    conf_score = 0.0
                    ratio = retracement

                    if 0.5 <= ratio <= 0.618:
                        conf_score += 30.0
                    elif 0.618 <= ratio <= 0.764:
                        conf_score += 25.0
                    elif 0.382 <= ratio <= 0.5:
                        conf_score += 20.0
                    else:
                        conf_score += 10.0

                    has_ob = not pd.isna(ob_high.iloc[i])
                    has_fvg = not pd.isna(fvg_high.iloc[i])

                    if has_ob:
                        conf_score += 15.0
                    if has_fvg:
                        conf_score += 10.0

                    if df['volume'].iloc[i] > vol_sma20.iloc[i] * 1.5:
                        conf_score += 15.0
                    elif df['volume'].iloc[i] > vol_sma20.iloc[i]:
                        conf_score += 10.0

                    if df['close'].iloc[i] > ema50.iloc[i]:
                        conf_score += 10.0

                    if useEWO:
                        if ewoBull.iloc[i]:
                            conf_score += 10.0
                        if ewoStrongL.iloc[i]:
                            conf_score += 15.0

                    # Cooldown check
                    if i - last_signal_bar < signalCooldown:
                        continue

                    # Direction change check
                    if requireDirectionChange and last_direction == 1:
                        continue

                    # Confluence score check
                    if conf_score < minConfScore:
                        continue

                    # Long Entry
                    if tradeDir in ["Both", "Long Only"]:
                        ewo_ok = not requireEWOConfirm or ewoBull.iloc[i] or ewoStrongL.iloc[i]
                        if ewo_ok:
                            trade_num += 1
                            entry_ts = int(df['time'].iloc[i])
                            entry_price = float(df['close'].iloc[i])
                            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()

                            entries.append({
                                'trade_num': trade_num,
                                'direction': 'long',
                                'entry_ts': entry_ts,
                                'entry_time': entry_time,
                                'entry_price_guess': entry_price,
                                'exit_ts': 0,
                                'exit_time': '',
                                'exit_price_guess': 0.0,
                                'raw_price_a': entry_price,
                                'raw_price_b': entry_price
                            })

                            last_signal_bar = i
                            last_direction = 1

                wave_size = abs(w1 - w0)
                if wave_size != 0:
                    retracement = (w0 - w2) / wave_size

                    if is_fib_valid(retracement):
                        conf_score = 0.0
                        ratio = retracement

                        if 0.5 <= ratio <= 0.618:
                            conf_score += 30.0
                        elif 0.618 <= ratio <= 0.764:
                            conf_score += 25.0
                        elif 0.382 <= ratio <= 0.5:
                            conf_score += 20.0
                        else:
                            conf_score += 10.0

                        has_ob = not pd.isna(ob_high.iloc[i])
                        has_fvg = not pd.isna(fvg_high.iloc[i])

                        if has_ob:
                            conf_score += 15.0
                        if has_fvg:
                            conf_score += 10.0

                        if df['volume'].iloc[i] > vol_sma20.iloc[i] * 1.5:
                            conf_score += 15.0
                        elif df['volume'].iloc[i] > vol_sma20.iloc[i]:
                            conf_score += 10.0

                        if df['close'].iloc[i] < ema50.iloc[i]:
                            conf_score += 10.0

                        if useEWO:
                            if ewoBear.iloc[i]:
                                conf_score += 10.0
                            if ewoStrongS.iloc[i]:
                                conf_score += 15.0

                        if i - last_signal_bar < signalCooldown:
                            continue

                        if requireDirectionChange and last_direction == -1:
                            continue

                        if conf_score < minConfScore:
                            continue

                        if tradeDir in ["Both", "Short Only"]:
                            ewo_ok = not requireEWOConfirm or ewoBear.iloc[i] or ewoStrongS.iloc[i]
                            if ewo_ok:
                                trade_num += 1
                                entry_ts = int(df['time'].iloc[i])
                                entry_price = float(df['close'].iloc[i])
                                entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()

                                entries.append({
                                    'trade_num': trade_num,
                                    'direction': 'short',
                                    'entry_ts': entry_ts,
                                    'entry_time': entry_time,
                                    'entry_price_guess': entry_price,
                                    'exit_ts': 0,
                                    'exit_time': '',
                                    'exit_price_guess': 0.0,
                                    'raw_price_a': entry_price,
                                    'raw_price_b': entry_price
                                })

                                last_signal_bar = i
                                last_direction = -1

    return entries