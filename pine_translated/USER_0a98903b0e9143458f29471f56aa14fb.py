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
    n = len(df)

    PeriodE2PSS = 15
    PriceE2PSS = (df['high'] + df['low']) / 2

    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3

    Filt2 = np.zeros(n)
    Filt2[0] = PriceE2PSS.iloc[0]
    Filt2[1] = PriceE2PSS.iloc[1]
    for i in range(2, n):
        Filt2[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
    Filt2_series = pd.Series(Filt2, index=df.index)
    TriggerE2PSS = pd.Series(Filt2).shift(1)

    baselineLong = Filt2_series > TriggerE2PSS
    baselineShort = Filt2_series < TriggerE2PSS

    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0

    pct_change = (close.diff(trendilo_smooth) / close * 100).fillna(0)
    
    def alma_approx(series, length, offset, sigma):
        m = (length - 1) * offset
        s = length / sigma
        w = np.exp(-np.square(np.arange(length) - m) / (2 * s * s))
        w /= w.sum()
        return series.rolling(length).apply(lambda x: np.sum(w * x), raw=True)

    avg_pct_change = alma_approx(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    avg_pct_change = pd.Series(avg_pct_change, index=df.index).fillna(0)
    rms = trendilo_bmult * np.sqrt((avg_pct_change ** 2).rolling(trendilo_length).sum() / trendilo_length)
    rms = pd.Series(rms, index=df.index).fillna(0)
    trendilo_dir = pd.Series(0, index=df.index)
    trendilo_dir[avg_pct_change > rms] = 1
    trendilo_dir[avg_pct_change < -rms] = -1

    trendiloLong = trendilo_dir == 1
    trendiloShort = trendilo_dir == -1

    length_TTMS = 20
    BB_mult_TTMS = 2.0
    KC_mult_low_TTMS = 2.0
    requireNoSqueeze = True

    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = BB_mult_TTMS * close.rolling(length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS

    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    devKC_TTMS = (high - low).rolling(length_TTMS).mean()
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS

    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)

    def linreg(series, length, offset):
        x = np.arange(length)
        x_mean = (length - 1) / 2
        sum_x = x.sum()
        sum_x2 = (x ** 2).sum()
        denom = length * sum_x2 - sum_x ** 2
        
        result = pd.Series(index=df.index, dtype=float)
        for i in range(length - 1, n):
            window = series.iloc[i - length + 1:i + 1].values
            y_mean = window.mean()
            sum_xy = np.sum(x * window)
            beta = (length * sum_xy - sum_x * y_mean) / denom
            alpha = y_mean - beta * x_mean
            result.iloc[i] = alpha + beta * offset
        return result

    avg_highest_lowest = (high.rolling(length_TTMS).max() + low.rolling(length_TTMS).min()) / 2
    avg_close_sma = close.rolling(length_TTMS).mean()
    mom_input = close - (avg_highest_lowest + avg_close_sma) / 2
    mom_TTMS = linreg(mom_input, length_TTMS, 0)

    momLong = (mom_TTMS > 0) & (mom_TTMS > mom_TTMS.shift(1))
    momShort = (mom_TTMS < 0) & (mom_TTMS < mom_TTMS.shift(1))

    ttmLong = momLong & (NoSqz_TTMS if requireNoSqueeze else True)
    ttmShort = momShort & (NoSqz_TTMS if requireNoSqueeze else True)

    RSI_Period = 6
    SF = 6
    QQE = 3
    ThreshHold = 3

    Wilders_Period = RSI_Period * 2 - 1

    def wilder_rsi(src, length):
        delta = src.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    Rsi = wilder_rsi(close, RSI_Period)
    RsiMa = Rsi.ewm(span=SF, adjust=False).mean()
    AtrRsi = np.abs(RsiMa.shift(1) - RsiMa)
    MaAtrRsi = AtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean()
    dar = MaAtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean() * QQE

    RSIndex = RsiMa
    newshortband = RSIndex + dar
    newlongband = RSIndex - dar

    longband = pd.Series(0.0, index=df.index)
    shortband = pd.Series(0.0, index=df.index)
    trend = pd.Series(1, index=df.index)

    for i in range(1, n):
        prev_lb = longband.iloc[i-1]
        prev_sb = shortband.iloc[i-1]
        rs = RSIndex.iloc[i]
        rs1 = RSIndex.iloc[i-1]
        
        if rs1 > prev_lb and rs > prev_lb:
            longband.iloc[i] = max(prev_lb, newlongband.iloc[i])
        else:
            longband.iloc[i] = newlongband.iloc[i]
            
        if rs1 < prev_sb and rs < prev_sb:
            shortband.iloc[i] = min(prev_sb, newshortband.iloc[i])
        else:
            shortband.iloc[i] = newshortband.iloc[i]

    cross_1 = (longband.shift(1) < RSIndex) & (longband > RSIndex)
    trend = pd.where((RSIndex < shortband.shift(1)) & (RSIndex > shortband), 1,
                     pd.where(cross_1, -1, trend.shift(1).fillna(1)))
    FastAtrRsiTL = pd.Series(np.where(trend == 1, longband, shortband), index=df.index)

    length_bb = 50
    qqeMult = 0.35
    basis = (FastAtrRsiTL - 50).rolling(length_bb).mean()
    dev = qqeMult * (FastAtrRsiTL - 50).rolling(length_bb).std()
    upper = basis + dev
    lower = basis - dev

    RSI_Period2 = 6
    SF2 = 5
    QQE2 = 1.61
    ThreshHold2 = 3

    Wilders_Period2 = RSI_Period2 * 2 - 1
    Rsi2 = wilder_rsi(close, RSI_Period2)
    RsiMa2 = Rsi2.ewm(span=SF2, adjust=False).mean()
    AtrRsi2 = np.abs(RsiMa2.shift(1) - RsiMa2)
    MaAtrRsi2 = AtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean()
    dar2 = MaAtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean() * QQE2

    RSIndex2 = RsiMa2
    newshortband2 = RSIndex2 + dar2
    newlongband2 = RSIndex2 - dar2

    longband2 = pd.Series(0.0, index=df.index)
    shortband2 = pd.Series(0.0, index=df.index)

    for i in range(1, n):
        prev_lb2 = longband2.iloc[i-1]
        prev_sb2 = shortband2.iloc[i-1]
        rs2 = RSIndex2.iloc[i]
        rs2_1 = RSIndex2.iloc[i-1]
        
        if rs2_1 > prev_lb2 and rs2 > prev_lb2:
            longband2.iloc[i] = max(prev_lb2, newlongband2.iloc[i])
        else:
            longband2.iloc[i] = newlongband2.iloc[i]
            
        if rs2_1 < prev_sb2 and rs2 < prev_sb2:
            shortband2.iloc[i] = min(prev_sb2, newshortband2.iloc[i])
        else:
            shortband2.iloc[i] = newshortband2.iloc[i]

    Greenbar1 = RsiMa2 - 50 > ThreshHold2
    Greenbar2 = RsiMa - 50 > upper
    Redbar1 = RsiMa2 - 50 < -ThreshHold2
    Redbar2 = RsiMa - 50 < lower

    qqeLong = Greenbar1 & Greenbar2
    qqeShort = Redbar1 & Redbar2

    useVolumeFilter = False
    vol_length = 50
    vol_threshold = 75

    vol_avg = volume.rolling(vol_length).mean()
    vol_normalized = volume / vol_avg * 100
    volumeOK = vol_normalized > vol_threshold if useVolumeFilter else True

    requireAllConfirmations = True

    longConfirmations = (trendiloLong.astype(int)) + (ttmLong.astype(int)) + (qqeLong.astype(int))
    shortConfirmations = (trendiloShort.astype(int)) + (ttmShort.astype(int)) + (qqeShort.astype(int))

    if requireAllConfirmations:
        long_condition = baselineLong & trendiloLong & ttmLong & qqeLong & volumeOK
        short_condition = baselineShort & trendiloShort & ttmShort & qqeShort & volumeOK
    else:
        long_condition = baselineLong & (longConfirmations >= 2) & volumeOK
        short_condition = baselineShort & (shortConfirmations >= 2) & volumeOK

    entries = []
    trade_num = 1

    for i in range(n):
        if np.isnan(Filt2[i]) or np.isnan(avg_pct_change.iloc[i]) or np.isnan(mom_TTMS.iloc[i]) if not np.isnan(mom_TTMS.iloc[i]) else True:
            if np.isnan(Rsi.iloc[i]):
                continue

        if long_condition.iloc[i]:
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
        elif short_condition.iloc[i]:
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