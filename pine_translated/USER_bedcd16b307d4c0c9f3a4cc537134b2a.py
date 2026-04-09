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
    
    # Input parameters
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
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # =====================
    # INDICATOR 1: EHLERS TWO POLE SUPER SMOOTHER
    # =====================
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * np.pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    PriceE2PSS = (df['high'] + df['low']) / 2
    
    Filt2 = np.zeros(len(df))
    TriggerE2PSS = np.zeros(len(df))
    Filt2[0] = PriceE2PSS.iloc[0]
    Filt2[1] = PriceE2PSS.iloc[1]
    TriggerE2PSS[0] = PriceE2PSS.iloc[0]
    
    for i in range(2, len(df)):
        Filt2[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
    
    TriggerE2PSS[1:] = Filt2[:-1]
    
    baselineLong = pd.Series(Filt2) > pd.Series(TriggerE2PSS)
    baselineShort = pd.Series(Filt2) < pd.Series(TriggerE2PSS)
    
    # =====================
    # INDICATOR 2: TRENDILO
    # =====================
    pct_change = close.pct_change(periods=trendilo_smooth) * 100
    
    def alma(src, length, offset, sigma):
        w = np.arange(length, dtype=float)
        w = np.exp(-(w - length * offset)**2 / (2 * sigma**2))
        w = w / w.sum()
        return pd.Series(src).rolling(length).apply(lambda x: np.dot(x, w[::-1]), raw=True).shift(1)
    
    avg_pct_change = alma(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    rms_vals = np.sqrt(pct_change.rolling(trendilo_length).apply(lambda x: (x**2).mean(), raw=True).shift(1))
    rms = trendilo_bmult * rms_vals
    
    trendiloLong = avg_pct_change > rms
    trendiloShort = avg_pct_change < -rms
    
    # =====================
    # INDICATOR 3: TTM SQUEEZE
    # =====================
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = close.rolling(length_TTMS).std() * BB_mult_TTMS
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TMS - dev_TTMS
    
    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    devKC_TTMS = tr.rolling(length_TTMS).mean()
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS
    
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    
    def linreg_slope(y, length):
        x = np.arange(length)
        x_mean = (length - 1) / 2
        y_mean = pd.Series(y).rolling(length).mean()
        numerator = pd.Series(y).rolling(length).apply(lambda y_vals: np.sum((x - x_mean) * (y_vals - y_vals.mean())), raw=True)
        denominator = np.sum((x - x_mean)**2)
        return numerator / denominator
    
    avg_high_low = (high.rolling(length_TTMS).max() + low.rolling(length_TTMS).min()) / 2
    avg_center = close.rolling(length_TTMS).mean()
    detrended = close - (avg_high_low + avg_center) / 2
    mom_TTMS = linreg_slope(detrended, length_TTMS)
    
    momLong = (mom_TTMS > 0) & (mom_TTMS > mom_TTMS.shift(1))
    momShort = (mom_TTMS < 0) & (mom_TTMS < mom_TTMS.shift(1))
    
    ttmLong = momLong & (NoSqz_TTMS if requireNoSqueeze else pd.Series(True, index=mom_TTMS.index))
    ttmShort = momShort & (NoSqz_TTMS if requireNoSqueeze else pd.Series(True, index=mom_TTMS.index))
    
    # =====================
    # INDICATOR 4: QQE MOD
    # =====================
    Wilders_Period = RSI_Period * 2 - 1
    
    def wilder_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    Rsi = wilder_rsi(close, RSI_Period)
    RsiMa = Rsi.ewm(span=SF, adjust=False).mean()
    AtrRsi = np.abs(RsiMa.shift(1) - RsiMa)
    MaAtrRsi = AtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean()
    dar = MaAtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean() * QQE
    
    longband = np.zeros(len(df))
    shortband = np.zeros(len(df))
    trend_arr = np.zeros(len(df))
    FastAtrRsiTL = np.zeros(len(df))
    
    RSIndex = RsiMa.values
    dar_vals = dar.values
    
    for i in range(1, len(df)):
        newshortband = RSIndex[i] + dar_vals[i]
        newlongband = RSIndex[i] - dar_vals[i]
        
        if RSIndex[i-1] > longband[i-1] and RSIndex[i] > longband[i-1]:
            longband[i] = max(longband[i-1], newlongband)
        else:
            longband[i] = newlongband
        
        if RSIndex[i-1] < shortband[i-1] and RSIndex[i] < shortband[i-1]:
            shortband[i] = min(shortband[i-1], newshortband)
        else:
            shortband[i] = newshortband
        
        if i >= 2:
            cross_1 = longband[i-1] < RSIndex[i-1] and RSIndex[i] < longband[i-1]
            if RSIndex[i] > shortband[i-1] and RSIndex[i-1] <= shortband[i-1]:
                trend_arr[i] = 1
            elif cross_1:
                trend_arr[i] = -1
            else:
                trend_arr[i] = trend_arr[i-1]
        else:
            trend_arr[i] = 1
        
        FastAtrRsiTL[i] = longband[i] if trend_arr[i] == 1 else shortband[i]
    
    basis = (pd.Series(FastAtrRsiTL) - 50).rolling(length_bb).mean()
    dev_qqe = (pd.Series(FastAtrRsiTL) - 50).rolling(length_bb).std() * qqeMult
    upper = basis + dev_qqe
    lower = basis - dev_qqe
    
    # QQE2
    Wilders_Period2 = RSI_Period2 * 2 - 1
    Rsi2 = wilder_rsi(close, RSI_Period2)
    RsiMa2 = Rsi2.ewm(span=SF2, adjust=False).mean()
    AtrRsi2 = np.abs(RsiMa2.shift(1) - RsiMa2)
    MaAtrRsi2 = AtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean()
    dar2 = MaAtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean() * QQE2
    
    longband2 = np.zeros(len(df))
    shortband2 = np.zeros(len(df))
    
    RSIndex2 = RsiMa2.values
    dar2_vals = dar2.values
    
    for i in range(1, len(df)):
        newshortband2 = RSIndex2[i] + dar2_vals[i]
        newlongband2 = RSIndex2[i] - dar2_vals[i]
        
        if RSIndex2[i-1] > longband2[i-1] and RSIndex2[i] > longband2[i-1]:
            longband2[i] = max(longband2[i-1], newlongband2)
        else:
            longband2[i] = newlongband2
        
        if RSIndex2[i-1] < shortband2[i-1] and RSIndex2[i] < shortband2[i-1]:
            shortband2[i] = min(shortband2[i-1], newshortband2)
        else:
            shortband2[i] = newshortband2
    
    RsiMa2_shifted = RsiMa2 - 50
    RsiMa_shifted = RsiMa - 50
    
    Greenbar1 = RsiMa2_shifted > ThreshHold2
    Greenbar2 = RsiMa_shifted > upper
    Redbar1 = RsiMa2_shifted < -ThreshHold2
    Redbar2 = RsiMa_shifted < lower
    
    qqeLong = Greenbar1 & Greenbar2
    qqeShort = Redbar1 & Redbar2
    
    # =====================
    # VOLUME FILTER
    # =====================
    vol_avg = volume.rolling(vol_length).mean()
    vol_normalized = volume / vol_avg * 100
    volumeOK = vol_normalized > vol_threshold if useVolumeFilter else pd.Series(True, index=close.index)
    
    # =====================
    # ENTRY CONDITIONS
    # =====================
    longConfirmations = (trendiloLong.astype(int)) + (ttmLong.astype(int)) + (qqeLong.astype(int))
    shortConfirmations = (trendiloShort.astype(int)) + (ttmShort.astype(int)) + (qqeShort.astype(int))
    
    if requireAllConfirmations:
        long_condition = baselineLong & trendiloLong & ttmLong & qqeLong & volumeOK
        short_condition = baselineShort & trendiloShort & ttmShort & qqeShort & volumeOK
    else:
        long_condition = baselineLong & (longConfirmations >= 2) & volumeOK
        short_condition = baselineShort & (shortConfirmations >= 2) & volumeOK
    
    # =====================
    # GENERATE ENTRIES
    # =====================
    entries = []
    trade_num = 1
    in_position = False
    
    for i in range(len(df)):
        if pd.isna(Filt2[i]) or pd.isna(avg_pct_change.iloc[i]) if not pd.isna(avg_pct_change.iloc[i]) else True:
            if i > 0 and not pd.isna(Filt2[i]) and not pd.isna(avg_pct_change.iloc[i]):
                pass
            else:
                continue
        
        if long_condition.iloc[i] and not in_position:
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
            in_position = True
        elif short_condition.iloc[i] and not in_position:
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
            in_position = True
        else:
            in_position = False
    
    return entries