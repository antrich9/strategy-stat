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
    n = len(df)
    if n < 3:
        return []

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume_arr = df['volume'].values
    open_arr = df['open'].values

    # Default parameters from Pine Script
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

    # ========== INDICATOR 1: EHLERS TWO POLE SUPER SMOOTHER ==========
    PriceE2PSS = (df['high'].values + df['low'].values) / 2  # hl2

    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3

    Filt2 = np.zeros(n)
    Filt2[0] = PriceE2PSS[0]
    Filt2[1] = PriceE2PSS[1]
    for i in range(2, n):
        Filt2[i] = coef1 * PriceE2PSS[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]

    TriggerE2PSS = np.roll(Filt2, 1)
    TriggerE2PSS[0] = PriceE2PSS[0]

    baselineLong = Filt2 > TriggerE2PSS
    baselineShort = Filt2 < TriggerE2PSS

    # ========== INDICATOR 2: TRENDILO ==========
    pct_change = np.zeros(n)
    for i in range(trendilo_smooth, n):
        pct_change[i] = (close[i] - close[i - trendilo_smooth]) / close[i - trendilo_smooth] * 100

    def alma(arr, length, offset, sigma):
        result = np.zeros(len(arr))
        window = np.arange(length)
        wts = np.exp(-((window - offset * (length - 1)) ** 2) / (2 * sigma ** 2))
        wts = wts / wts.sum()
        for i in range(length - 1, len(arr)):
            result[i] = np.sum(arr[i - length + 1:i + 1] * wts)
        return result

    avg_pct_change = alma(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)

    rms = np.zeros(n)
    for i in range(trendilo_length - 1, n):
        rms[i] = trendilo_bmult * np.sqrt(np.mean(avg_pct_change[i - trendilo_length + 1:i + 1] ** 2))

    trendilo_dir = np.where(avg_pct_change > rms, 1, np.where(avg_pct_change < -rms, -1, 0))
    trendiloLong = trendilo_dir == 1
    trendiloShort = trendilo_dir == -1

    # ========== INDICATOR 3: TTM SQUEEZE ==========
    BB_basis_TTMS = pd.Series(close).rolling(length_TTMS).mean().values
    close_std = pd.Series(close).rolling(length_TTMS).std().values
    dev_TTMS = BB_mult_TTMS * close_std
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS

    KC_basis_TTMS = BB_basis_TTMS.copy()
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], max(abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
    devKC_TTMS = pd.Series(tr).rolling(length_TTMS).mean().values
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS

    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)

    highest_high = pd.Series(high).rolling(length_TTMS).max().values
    lowest_low = pd.Series(low).rolling(length_TTMS).min().values
    sma_close = BB_basis_TTMS.copy()
    avg_val = (highest_high + lowest_low) / 2 + sma_close
    mom_component = close - avg_val

    mom_TTMS = np.zeros(n)
    for i in range(length_TTMS - 1, n):
        x = np.arange(length_TTMS)
        y = mom_component[i - length_TTMS + 1:i + 1]
        if len(y) == length_TTMS:
            slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
            intercept = y.mean() - slope * x.mean()
            mom_TTMS[i] = slope * (length_TTMS - 1) + intercept

    momLong = (mom_TTMS > 0) & (mom_TTMS > np.roll(mom_TTMS, 1))
    momShort = (mom_TTMS < 0) & (mom_TTMS < np.roll(mom_TTMS, 1))
    momLong[0] = False
    momShort[0] = False

    ttmLong = momLong & (NoSqz_TTMS | ~requireNoSqueeze)
    ttmShort = momShort & (NoSqz_TTMS | ~requireNoSqueeze)

    # ========== INDICATOR 4: QQE MOD ==========
    Wilders_Period = RSI_Period * 2 - 1

    def wilder_rsi(series, period):
        delta = np.diff(series, prepend=series[0])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros(len(series))
        avg_loss = np.zeros(len(series))
        if len(series) >= period:
            avg_gain[period - 1] = np.mean(gains[:period])
            avg_loss[period - 1] = np.mean(losses[:period])
            for i in range(period, len(series)):
                avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
                avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period
        return avg_gain / (avg_loss + 1e-10) * 100

    Rsi = wilder_rsi(close, RSI_Period)
    RsiMa = pd.Series(Rsi).ewm(span=SF, adjust=False).mean().values
    AtrRsi = np.abs(pd.Series(RsiMa).shift(1).fillna(0).values - RsiMa)
    MaAtrRsi = pd.Series(AtrRsi).ewm(span=Wilders_Period, adjust=False).mean().values
    dar = pd.Series(MaAtrRsi).ewm(span=Wilders_Period, adjust=False).mean().values * QQE

    longband = np.zeros(n)
    shortband = np.zeros(n)
    trend = np.ones(n)

    RSIndex = RsiMa.copy()
    newshortband = RSIndex + dar
    newlongband = RSIndex - dar

    for i in range(1, n):
        if RSIndex[i-1] > longband[i-1] and RSIndex[i] > longband[i-1]:
            longband[i] = max(longband[i-1], newlongband[i])
        else:
            longband[i] = newlongband[i]
        if RSIndex[i-1] < shortband[i-1] and RSIndex[i] < shortband[i-1]:
            shortband[i] = min(shortband[i-1], newshortband[i])
        else:
            shortband[i] = newshortband[i]

    longband_prev = np.roll(longband, 1)
    shortband_prev = np.roll(shortband, 1)
    RSIndex_prev = np.roll(RSIndex, 1)

    cross_1 = (longband_prev > RSIndex_prev) & (longband > RSIndex)
    trend_prev = np.roll(trend, 1, mode='wrap')
    trend_prev[0] = 1
    cross_short = (RSIndex_prev < shortband_prev) & (RSIndex < shortband)
    trend = np.where(cross_short, 1, np.where(cross_1, -1, trend_prev))

    FastAtrRsiTL = np.where(trend == 1, longband, shortband)

    basis = pd.Series(FastAtrRsiTL - 50).rolling(length_bb).mean().values
    dev = qqeMult * pd.Series(FastAtrRsiTL - 50).rolling(length_bb).std().values
    upper = basis + dev
    lower = basis - dev

    Wilders_Period2 = RSI_Period2 * 2 - 1
    Rsi2 = wilder_rsi(close, RSI_Period2)
    RsiMa2 = pd.Series(Rsi2).ewm(span=SF2, adjust=False).mean().values
    AtrRsi2 = np.abs(pd.Series(RsiMa2).shift(1).fillna(0).values - RsiMa2)
    MaAtrRsi2 = pd.Series(AtrRsi2).ewm(span=Wilders_Period2, adjust=False).mean().values
    dar2 = pd.Series(MaAtrRsi2).ewm(span=Wilders_Period2, adjust=False).mean().values * QQE2

    longband2 = np.zeros(n)
    shortband2 = np.zeros(n)

    RSIndex2 = RsiMa2.copy()
    newshortband2 = RSIndex2 + dar2
    newlongband2 = RSIndex2 - dar2

    for i in range(1, n):
        if RSIndex2[i-1] > longband2[i-1] and RSIndex2[i] > longband2[i-1]:
            longband2[i] = max(longband2[i-1], newlongband2[i])
        else:
            longband2[i] = newlongband2[i]
        if RSIndex2[i-1] < shortband2[i-1] and RSIndex2[i] < shortband2[i-1]:
            shortband2[i] = min(shortband2[i-1], newshortband2[i])
        else:
            shortband2[i] = newshortband2[i]

    Greenbar1 = RsiMa2 - 50 > ThreshHold2
    Greenbar2 = RsiMa - 50 > upper
    Redbar1 = RsiMa2 - 50 < -ThreshHold2
    Redbar2 = RsiMa - 50 < lower

    qqeLong = Greenbar1 & Greenbar2
    qqeShort = Redbar1 & Redbar2

    # ========== VOLUME FILTER ==========
    vol_avg = pd.Series(volume_arr).rolling(vol_length).mean().values
    vol_normalized = volume_arr / vol_avg * 100
    volumeOK = vol_normalized > vol_threshold if useVolumeFilter else np.ones(n, dtype=bool)

    # ========== COMBINED ENTRY LOGIC ==========
    longConfirmations = (trendiloLong.astype(int)) + (ttmLong.astype(int)) + (qqeLong.astype(int))
    shortConfirmations = (trendiloShort.astype(int)) + (ttmShort.astype(int)) + (qqeShort.astype(int))

    long_condition = np.zeros(n, dtype=bool)
    short_condition = np.zeros(n, dtype=bool)

    if requireAllConfirmations:
        long_condition = baselineLong & trendiloLong & ttmLong & qqeLong & volumeOK
        short_condition = baselineShort & trendiloShort & ttmShort & qqeShort & volumeOK
    else:
        long_condition = baselineLong & (longConfirmations >= 2) & volumeOK
        short_condition = baselineShort & (shortConfirmations >= 2) & volumeOK

    # ========== GENERATE ENTRIES ==========
    entries = []
    trade_num = 1

    for i in range(1, n):
        if i < 3:
            continue
        if np.isnan(Filt2[i]) or np.isnan(avg_pct_change[i]) or np.isnan(mom_TTMS[i]) or np.isnan(Rsi[i]) or np.isnan(Rsi2[i]):
            continue

        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = float(close[i])

        if long_condition[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
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

        if short_condition[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
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