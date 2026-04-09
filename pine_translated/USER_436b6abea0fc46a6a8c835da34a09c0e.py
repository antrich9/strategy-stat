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
    time_col = df['time']

    # Parameters
    PeriodE2PSS = 15
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    KC_mult_low_TTMS = 2.0
    requireNoSqueeze = True
    RSI_Period = 6
    SF = 6
    QQE = 3
    ThreshHold = 3
    length_bb = 50
    qqeMult = 0.35
    RSI_Period2 = 6
    SF2 = 5
    QQE2 = 1.61
    ThreshHold2 = 3
    useVolumeFilter = False
    vol_length = 50
    vol_threshold = 75
    requireAllConfirmations = True

    # Ehlers Two Pole Super Smoother
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * np.pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    PriceE2PSS = (high + low) / 2

    filt2 = pd.Series(index=df.index, dtype=float)
    trigger = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i < 2:
            filt2.iloc[i] = PriceE2PSS.iloc[i]
        else:
            filt2.iloc[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * filt2.iloc[i-1] + coef3 * filt2.iloc[i-2]
        if i > 0:
            trigger.iloc[i] = filt2.iloc[i-1]
        else:
            trigger.iloc[i] = PriceE2PSS.iloc[i]

    baselineLong = filt2 > trigger
    baselineShort = filt2 < trigger

    # Trendilo
    pct_change = close.diff(trendilo_smooth) / close * 100
    def alma_calc(series, length, offset, sigma):
        result = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            if i < length:
                result.iloc[i] = np.nan
            else:
                window = series.iloc[i-length+1:i+1].values
                k = np.arange(length)
                w = np.exp(-((k - offset * (length - 1))**2) / (2 * sigma**2))
                w = w / np.sum(w)
                result.iloc[i] = np.dot(window, w)
        return result

    avg_pct_change = alma_calc(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    rms_vals = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i < trendilo_length:
            rms_vals.iloc[i] = np.nan
        else:
            window = avg_pct_change.iloc[i-trendilo_length+1:i+1].values
            rms_vals.iloc[i] = trendilo_bmult * np.sqrt(np.mean(window**2))
    trendilo_dir = pd.Series(index=df.index, dtype=int)
    for i in range(len(df)):
        if pd.isna(avg_pct_change.iloc[i]) or pd.isna(rms_vals.iloc[i]):
            trendilo_dir.iloc[i] = 0
        elif avg_pct_change.iloc[i] > rms_vals.iloc[i]:
            trendilo_dir.iloc[i] = 1
        elif avg_pct_change.iloc[i] < -rms_vals.iloc[i]:
            trendilo_dir.iloc[i] = -1
        else:
            trendilo_dir.iloc[i] = 0
    trendiloLong = trendilo_dir == 1
    trendiloShort = trendilo_dir == -1

    # TTM Squeeze
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    close_std = close.rolling(length_TTMS).std()
    dev_TTMS = BB_mult_TTMS * close_std
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS
    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    tr = pd.concat([high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1))], axis=1).max(axis=1)
    devKC_TTMS = tr.rolling(length_TTMS).mean()
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)

    mom_TTMS = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i < length_TTMS:
            mom_TTMS.iloc[i] = np.nan
        else:
            highest_high = high.iloc[i-length_TTMS+1:i+1].max()
            lowest_low = low.iloc[i-length_TTMS+1:i+1].min()
            avg_price = np.mean([highest_high, lowest_low, BB_basis_TTMS.iloc[i]])
            dev_vals = close.iloc[i-length_TTMS+1:i+1] - avg_price
            x = np.arange(length_TTMS)
            slope = np.polyfit(x, dev_vals, 1)[0]
            mom_TTMS.iloc[i] = slope

    momLong = (mom_TTMS > 0) & (mom_TTMS > mom_TTMS.shift(1))
    momShort = (mom_TTMS < 0) & (mom_TTMS < mom_TTMS.shift(1))
    ttmLong = momLong & (NoSqz_TTMS if requireNoSqueeze else True)
    ttmShort = momShort & (NoSqz_TTMS if requireNoSqueeze else True)

    # QQE MOD
    Wilders_Period = RSI_Period * 2 - 1

    def wilder_rsi(src, period):
        delta = src.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val

    Rsi = wilder_rsi(close, RSI_Period)
    RsiMa = Rsi.ewm(span=SF, adjust=False).mean()
    AtrRsi = np.abs(RsiMa.shift(1) - RsiMa)
    MaAtrRsi = AtrRsi.ewm(alpha=1/Wilders_Period, min_periods=Wilders_Period, adjust=False).mean()
    dar = MaAtrRsi * QQE

    longband = pd.Series(0.0, index=df.index)
    shortband = pd.Series(0.0, index=df.index)
    trend = pd.Series(1, index=df.index)
    prev_longband = pd.Series(0.0, index=df.index)
    prev_shortband = pd.Series(0.0, index=df.index)

    for i in range(1, len(df)):
        prev_longband.iloc[i] = longband.iloc[i-1]
        prev_shortband.iloc[i] = shortband.iloc[i-1]
        new_longband = RsiMa.iloc[i] - dar.iloc[i]
        new_shortband = RsiMa.iloc[i] + dar.iloc[i]
        if RsiMa.iloc[i-1] > longband.iloc[i-1] and RsiMa.iloc[i] > longband.iloc[i-1]:
            longband.iloc[i] = max(longband.iloc[i-1], new_longband)
        else:
            longband.iloc[i] = new_longband
        if RsiMa.iloc[i-1] < shortband.iloc[i-1] and RsiMa.iloc[i] < shortband.iloc[i-1]:
            shortband.iloc[i] = min(shortband.iloc[i-1], new_shortband)
        else:
            shortband.iloc[i] = new_shortband

    for i in range(1, len(df)):
        cross_long = prev_longband.iloc[i] > RsiMa.iloc[i-1] and RsiMa.iloc[i] <= prev_longband.iloc[i]
        cross_short = prev_shortband.iloc[i] < RsiMa.iloc[i-1] and RsiMa.iloc[i] >= prev_shortband.iloc[i]
        if cross_short:
            trend.iloc[i] = 1
        elif cross_long:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1] if i > 0 else 1

    FastAtrRsiTL = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        FastAtrRsiTL.iloc[i] = longband.iloc[i] if trend.iloc[i] == 1 else shortband.iloc[i]

    # QQE2
    Wilders_Period2 = RSI_Period2 * 2 - 1
    Rsi2 = wilder_rsi(close, RSI_Period2)
    RsiMa2 = Rsi2.ewm(span=SF2, adjust=False).mean()
    AtrRsi2 = np.abs(RsiMa2.shift(1) - RsiMa2)
    MaAtrRsi2 = AtrRsi2.ewm(alpha=1/Wilders_Period2, min_periods=Wilders_Period2, adjust=False).mean()
    dar2 = MaAtrRsi2 * QQE2

    longband2 = pd.Series(0.0, index=df.index)
    shortband2 = pd.Series(0.0, index=df.index)

    for i in range(1, len(df)):
        new_longband2 = RsiMa2.iloc[i] - dar2.iloc[i]
        new_shortband2 = RsiMa2.iloc[i] + dar2.iloc[i]
        if RsiMa2.iloc[i-1] > longband2.iloc[i-1] and RsiMa2.iloc[i] > longband2.iloc[i-1]:
            longband2.iloc[i] = max(longband2.iloc[i-1], new_longband2)
        else:
            longband2.iloc[i] = new_longband2
        if RsiMa2.iloc[i-1] < shortband2.iloc[i-1] and RsiMa2.iloc[i] < shortband2.iloc[i-1]:
            shortband2.iloc[i] = min(shortband2.iloc[i-1], new_shortband2)
        else:
            shortband2.iloc[i] = new_shortband2

    qqe_basis = (FastAtrRsiTL - 50).rolling(length_bb).mean()
    qqe_std = (FastAtrRsiTL - 50).rolling(length_bb).std()
    qqe_upper = qqe_basis + qqeMult * qqe_std
    qqe_lower = qqe_basis - qqeMult * qqe_std

    Greenbar1 = RsiMa2 - 50 > ThreshHold2
    Greenbar2 = RsiMa - 50 > qqe_upper
    Redbar1 = RsiMa2 - 50 < -ThreshHold2
    Redbar2 = RsiMa - 50 < qqe_lower

    qqeLong = Greenbar1 & Greenbar2
    qqeShort = Redbar1 & Redbar2

    # Volume Filter
    vol_avg = volume.rolling(vol_length).mean()
    vol_normalized = volume / vol_avg * 100
    volumeOK = vol_normalized > vol_threshold if useVolumeFilter else pd.Series(True, index=df.index)

    # Entry conditions
    longConfirmations = trendiloLong.astype(int) + ttmLong.astype(int) + qqeLong.astype(int)
    shortConfirmations = trendiloShort.astype(int) + ttmShort.astype(int) + qqeShort.astype(int)

    if requireAllConfirmations:
        long_condition = baselineLong & trendiloLong & ttmLong & qqeLong & volumeOK
        short_condition = baselineShort & trendiloShort & ttmShort & qqeShort & volumeOK
    else:
        long_condition = baselineLong & (longConfirmations >= 2) & volumeOK
        short_condition = baselineShort & (shortConfirmations >= 2) & volumeOK

    # Build entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(filt2.iloc[i]) or pd.isna(trendiloLong.iloc[i]) or pd.isna(ttmLong.iloc[i]) or pd.isna(qqeLong.iloc[i]):
            continue
        if long_condition.iloc[i]:
            ts = int(time_col.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        if short_condition.iloc[i]:
            ts = int(time_col.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1

    return entries