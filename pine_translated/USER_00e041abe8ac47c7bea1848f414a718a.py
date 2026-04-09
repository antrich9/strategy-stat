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

    # Ehlers Two Pole Super Smoother
    PeriodE2PSS = 15
    pi = 2 * np.arcsin(1.0)
    a1 = np.exp(-1.414 * np.pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3

    PriceE2PSS = (high + low) / 2.0
    Filt2 = pd.Series(np.nan, index=df.index)
    Filt2.iloc[0] = PriceE2PSS.iloc[0]
    Filt2.iloc[1] = PriceE2PSS.iloc[1]
    Filt2.iloc[2] = PriceE2PSS.iloc[2]

    for i in range(3, len(df)):
        Filt2.iloc[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * Filt2.iloc[i-1] + coef3 * Filt2.iloc[i-2]

    TriggerE2PSS = Filt2.shift(1)
    signalLongE2PSS = Filt2 > TriggerE2PSS
    signalShortE2PSS = Filt2 < TriggerE2PSS

    # QQE Mod
    RSI_Period = 6
    SF = 6
    QQE = 3
    ThreshHold = 3
    Wilders_Period = RSI_Period * 2 - 1

    def wilder_rsi(series, period):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    Rsi = wilder_rsi(close, RSI_Period)
    RsiMa = Rsi.ewm(span=SF, adjust=False).mean()
    AtrRsi = (RsiMa.shift(1) - RsiMa).abs()
    MaAtrRsi = AtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean()
    dar = MaAtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean() * QQE

    RSIndex = RsiMa
    newshortband = RSIndex + dar
    newlongband = RSIndex - dar

    longband = pd.Series(0.0, index=df.index)
    shortband = pd.Series(0.0, index=df.index)
    trend = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        if RSIndex.iloc[i-1] > longband.iloc[i-1] and RSIndex.iloc[i] > longband.iloc[i-1]:
            longband.iloc[i] = max(longband.iloc[i-1], newlongband.iloc[i])
        else:
            longband.iloc[i] = newlongband.iloc[i]
        if RSIndex.iloc[i-1] < shortband.iloc[i-1] and RSIndex.iloc[i] < shortband.iloc[i-1]:
            shortband.iloc[i] = min(shortband.iloc[i-1], newshortband.iloc[i])
        else:
            shortband.iloc[i] = newshortband.iloc[i]
        if ta_cross(RSIndex, shortband.shift(1), i):
            trend.iloc[i] = 1
        elif ta_cross(longband.shift(1), RSIndex, i):
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1] if i > 0 else 1

    FastAtrRsiTL = pd.Series(np.where(trend == 1, longband, shortband), index=df.index)

    # QQE Mod #2 with Bollinger
    length = 50
    qqeMult = 0.35
    basis = (FastAtrRsiTL - 50).rolling(length).mean()
    dev = qqeMult * (FastAtrRsiTL - 50).rolling(length).std()
    upper = basis + dev
    lower = basis - dev

    RSI_Period2 = 6
    SF2 = 5
    QQE2 = 1.61
    ThreshHold2 = 3
    Wilders_Period2 = RSI_Period2 * 2 - 1

    Rsi2 = wilder_rsi(close, RSI_Period2)
    RsiMa2 = Rsi2.ewm(span=SF2, adjust=False).mean()
    AtrRsi2 = (RsiMa2.shift(1) - RsiMa2).abs()
    MaAtrRsi2 = AtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean()
    dar2 = MaAtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean() * QQE2

    RSIndex2 = RsiMa2
    newshortband2 = RSIndex2 + dar2
    newlongband2 = RSIndex2 - dar2

    longband2 = pd.Series(0.0, index=df.index)
    shortband2 = pd.Series(0.0, index=df.index)
    trend2 = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        if RSIndex2.iloc[i-1] > longband2.iloc[i-1] and RSIndex2.iloc[i] > longband2.iloc[i-1]:
            longband2.iloc[i] = max(longband2.iloc[i-1], newlongband2.iloc[i])
        else:
            longband2.iloc[i] = newlongband2.iloc[i]
        if RSIndex2.iloc[i-1] < shortband2.iloc[i-1] and RSIndex2.iloc[i] < shortband2.iloc[i-1]:
            shortband2.iloc[i] = min(shortband2.iloc[i-1], newshortband2.iloc[i])
        else:
            shortband2.iloc[i] = newshortband2.iloc[i]
        if ta_cross(RSIndex2, shortband2.shift(1), i):
            trend2.iloc[i] = 1
        elif ta_cross(longband2.shift(1), RSIndex2, i):
            trend2.iloc[i] = -1
        else:
            trend2.iloc[i] = trend2.iloc[i-1] if i > 0 else 1

    Greenbar1 = RsiMa2 - 50 > ThreshHold2
    Greenbar2 = RsiMa - 50 > upper
    Redbar1 = RsiMa2 - 50 < -ThreshHold2
    Redbar2 = RsiMa - 50 < lower

    qqeBuy = Greenbar1 & Greenbar2
    qqeSell = Redbar1 & Redbar2

    # Trendilo
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6

    pct_change = close.diff(trendilo_smooth) / close * 100

    def alma(data, length, offset, sigma):
        window = np.arange(length)
        m = offset * (length - 1)
        s = sigma * length / 6.0
        w = np.exp(-((window - m) ** 2) / (2 * s * s))
        w = w / w.sum()
        result = pd.Series(np.nan, index=data.index)
        for i in range(length - 1, len(data)):
            window_data = data.iloc[i - length + 1:i + 1]
            if window_data.notna().all():
                result.iloc[i] = np.sum(w * window_data.values)
        return result

    avg_pct_change = alma(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    rms = np.sqrt((avg_pct_change ** 2).rolling(trendilo_length).sum() / trendilo_length) * 1.0

    trendilo_dir = pd.Series(0, index=df.index)
    trendilo_dir[avg_pct_change > rms] = 1
    trendilo_dir[avg_pct_change < -rms] = -1

    trendiloBuy = trendilo_dir == 1
    trendiloSell = trendilo_dir == -1

    # TTM Squeeze
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    KC_mult_low_TTMS = 2.0

    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = BB_mult_TTMS * close.rolling(length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS

    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    devKC_TTMS = ((high - low).clip(lower=0)).rolling(length_TTMS).mean()
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS

    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)

    highest = high.rolling(length_TTMS).max()
    lowest = low.rolling(length_TTMS).min()
    avg_val = (highest + lowest) / 2 + close.rolling(length_TTMS).mean()
    mom_source = close - avg_val

    def rolling_linreg(y, length):
        x = np.arange(length)
        x_mean = (length - 1) / 2.0
        x_sum = x.sum()
        x_sq_sum = (x * x).sum()
        result = pd.Series(np.nan, index=y.index)
        for i in range(length - 1, len(y)):
            window = y.iloc[i - length + 1:i + 1]
            if window.notna().all():
                y_mean = window.mean()
                y_sum = window.sum()
                numerator = np.sum((x - x_mean) * (window.values - y_mean))
                denominator = x_sq_sum - x_sum * x_sum / length
                if denominator == 0:
                    result.iloc[i] = 0
                else:
                    slope = numerator / denominator
                    intercept = y_mean - slope * x_mean
                    result.iloc[i] = slope * (length - 1) + intercept
        return result

    mom_TTMS = rolling_linreg(mom_source, length_TTMS)

    ttmBuy = NoSqz_TTMS & (mom_TTMS > 0) & (mom_TTMS > mom_TTMS.shift(1))
    ttmSell = NoSqz_TTMS & (mom_TTMS < 0) & (mom_TTMS < mom_TTMS.shift(1))

    # Volume Filter
    vol_length = 50
    vol_threshold = 75

    vol_ma = volume.rolling(vol_length).mean()
    vol_std = volume.rolling(vol_length).std()
    norm_vol = (volume - vol_ma) / vol_std * 100 + 100

    volFilter = norm_vol > vol_threshold

    # SSL Hybrid Baseline
    ssl_len = 60

    def wma(src, length):
        return src.rolling(window=length).apply(lambda x: np.sum(x * np.arange(1, length+1)) / np.sum(np.arange(1, length+1)), raw=True)

    def hma_calc(src, length):
        half = wma(src, int(length / 2))
        sqrt_len = int(np.round(np.sqrt(length)))
        return wma(2 * half - wma(src, length), sqrt_len)

    ssl_baseline = hma_calc(close, ssl_len)
    ssl_up = hma_calc(high, ssl_len)
    ssl_down = hma_calc(low, ssl_len)

    ssl_hlv = pd.Series(0, index=df.index)
    ssl_hlv[ssl_baseline > ssl_baseline.shift(1)] = 1
    ssl_hlv[ssl_baseline < ssl_baseline.shift(1)] = -1

    sslBuy = ssl_hlv > 0
    sslSell = ssl_hlv < 0

    # Combined entry logic
    longCondition = signalLongE2PSS & qqeBuy & trendiloBuy & ttmBuy & sslBuy & volFilter
    shortCondition = signalShortE2PSS & qqeSell & trendiloSell & ttmSell & sslSell & volFilter

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if longCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
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
        elif shortCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
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