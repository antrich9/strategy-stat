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
    open_prices = df['open']
    n = len(df)

    PeriodE2PSS = 15
    PriceE2PSS = (high + low) / 2
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3

    Filt2 = np.zeros(n)
    TriggerE2PSS = np.zeros(n)
    Filt2[0] = PriceE2PSS.iloc[0]
    Filt2[1] = PriceE2PSS.iloc[1]
    for i in range(2, n):
        Filt2[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
    TriggerE2PSS[1:] = Filt2[:-1]
    baselineLong = Filt2 > TriggerE2PSS
    baselineShort = Filt2 < TriggerE2PSS

    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    pct_change = close.diff(trendilo_smooth) / close * 100
    avg_pct_change = pct_change.ewm(span=trendilo_length, adjust=False).mean()
    rms_arr = np.sqrt((avg_pct_change ** 2).rolling(trendilo_length).mean())
    rms = trendilo_bmult * rms_arr
    trendilo_dir = np.where(avg_pct_change > rms, 1, np.where(avg_pct_change < -rms, -1, 0))
    trendiloLong = trendilo_dir == 1
    trendiloShort = trendilo_dir == -1

    length_TTMS = 20
    BB_mult_TTMS = 2.0
    KC_mult_low_TTMS = 2.0
    requireNoSqueeze = True
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = close.rolling(length_TTMS).std() * BB_mult_TTMS
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS
    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    devKC_TTMS = tr.rolling(length_TTMS).mean()
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    mom_TTMS_vals = close - (highest_high + lowest_low + close.rolling(length_TTMS).mean()) / 3
    mom_TTMS = mom_TTMS_vals.rolling(length_TTMS).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] * (length_TTMS - 1) / 2 + np.mean(x), raw=True)
    momLong_raw = (mom_TTMS > 0) & (mom_TTMS > mom_TTMS.shift(1))
    momShort_raw = (mom_TTMS < 0) & (mom_TTMS < mom_TTMS.shift(1))
    ttmLong_raw = momLong_raw & (NoSqz_TTMS if requireNoSqueeze else True)
    ttmShort_raw = momShort_raw & (NoSqz_TTMS if requireNoSqueeze else True)

    RSI_Period = 6
    SF = 6
    QQE = 3
    ThreshHold = 3
    Wilders_Period = RSI_Period * 2 - 1
    delta_close = close.diff()
    gain = delta_close.where(delta_close > 0, 0.0)
    loss = (-delta_close).where(delta_close < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/Wilders_Period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/Wilders_Period, adjust=False).mean()
    rs = avg_gain / avg_loss
    Rsi = 100 - (100 / (1 + rs))
    RsiMa = Rsi.ewm(span=SF, adjust=False).mean()
    AtrRsi = np.abs(RsiMa.shift(1) - RsiMa)
    MaAtrRsi = AtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean()
    dar = MaAtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean() * QQE
    longband = np.zeros(n)
    shortband = np.zeros(n)
    trend = np.zeros(n)
    RSIndex = RsiMa.values
    DeltaFastAtrRsi = dar.values
    newshortband = RSIndex + DeltaFastAtrRsi
    newlongband = RSIndex - DeltaFastAtrRsi
    for i in range(1, n):
        if RSIndex[i-1] > longband[i-1] and RSIndex[i] > longband[i-1]:
            longband[i] = max(longband[i-1], newlongband[i])
        else:
            longband[i] = newlongband[i]
        if RSIndex[i-1] < shortband[i-1] and RSIndex[i] < shortband[i-1]:
            shortband[i] = min(shortband[i-1], newshortband[i])
        else:
            shortband[i] = newshortband[i]
        if i > 0:
            if RSIndex[i] < shortband[i-1]:
                trend[i] = 1
            elif RSIndex[i-1] < longband[i-1] and RSIndex[i] > longband[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
        else:
            trend[i] = 1
    FastAtrRsiTL = np.where(trend == 1, longband, shortband)
    length_bb = 50
    qqeMult = 0.35
    FastAtrRsiTL_series = pd.Series(FastAtrRsiTL)
    basis = (FastAtrRsiTL_series - 50).rolling(length_bb).mean()
    dev_qqe = (FastAtrRsiTL_series - 50).rolling(length_bb).std() * qqeMult
    upper = basis + dev_qqe
    lower = basis - dev_qqe

    RSI_Period2 = 6
    SF2 = 5
    QQE2 = 1.61
    ThreshHold2 = 3
    Wilders_Period2 = RSI_Period2 * 2 - 1
    delta_close2 = close.diff()
    gain2 = delta_close2.where(delta_close2 > 0, 0.0)
    loss2 = (-delta_close2).where(delta_close2 < 0, 0.0)
    avg_gain2 = gain2.ewm(alpha=1/Wilders_Period2, adjust=False).mean()
    avg_loss2 = loss2.ewm(alpha=1/Wilders_Period2, adjust=False).mean()
    rs2 = avg_gain2 / avg_loss2
    Rsi2 = 100 - (100 / (1 + rs2))
    RsiMa2 = Rsi2.ewm(span=SF2, adjust=False).mean()
    AtrRsi2 = np.abs(RsiMa2.shift(1) - RsiMa2)
    MaAtrRsi2 = AtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean()
    dar2 = MaAtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean() * QQE2
    longband2 = np.zeros(n)
    shortband2 = np.zeros(n)
    RSIndex2 = RsiMa2.values
    newshortband2 = RSIndex2 + dar2.values
    newlongband2 = RSIndex2 - dar2.values
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
    qqeLong_raw = Greenbar1 & Greenbar2
    qqeShort_raw = Redbar1 & Redbar2

    longCond = baselineLong & trendiloLong & ttmLong_raw & qqeLong_raw
    shortCond = baselineShort & trendiloShort & ttmShort_raw & qqeShort_raw

    entries = []
    trade_num = 1

    for i in range(n):
        if pd.isna(Filt2[i]) or pd.isna(TriggerE2PSS[i]) or pd.isna(avg_pct_change.iloc[i]) if i < len(avg_pct_change) else True:
            continue
        if i < 2:
            continue
        if i < trendilo_length:
            continue
        if i < length_TTMS:
            continue
        if pd.isna(mom_TTMS.iloc[i]) if i < len(mom_TTMS) else True:
            continue
        if i < length_bb:
            continue
        if pd.isna(RsiMa.iloc[i]) or pd.isna(RsiMa2.iloc[i]):
            continue
        if pd.isna(upper.iloc[i]) if i < len(upper) else True:
            continue

        if longCond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

        if shortCond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

    return entries