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
    if n < 60:
        return []
    
    # Input settings
    use_QQE = True
    use_SSL = True
    use_WAE = True
    use_E2PSS = True
    use_Trendilo = True
    use_TTM = True
    use_ALMA = False
    use_QuickSilver = False
    entry_mode = "All Active"
    
    # ========== QQE MOD ==========
    RSI_Period = 6
    SF = 6
    QQE = 3
    ThreshHold = 3
    qqeSrc = close
    Wilders_Period = RSI_Period * 2 - 1
    
    def wilders_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    Rsi = wilders_rsi(qqeSrc, RSI_Period)
    RsiMa = Rsi.ewm(span=SF, adjust=False).mean()
    AtrRsi = (RsiMa.shift(1) - RsiMa).abs()
    MaAtrRsi = AtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean()
    dar = MaAtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean() * QQE
    
    longband = pd.Series(0.0, index=df.index)
    shortband = pd.Series(0.0, index=df.index)
    trend = pd.Series(1, index=df.index)
    FastAtrRsiTL = pd.Series(0.0, index=df.index)
    
    length_bb = 50
    qqeMult = 0.35
    
    for i in range(1, n):
        RSIndex = RsiMa.iloc[i]
        DeltaFastAtrRsi = dar.iloc[i]
        newshortband_val = RSIndex + DeltaFastAtrRsi
        newlongband_val = RSIndex - DeltaFastAtrRsi
        
        prev_longband = longband.iloc[i-1]
        prev_shortband = shortband.iloc[i-1]
        prev_trend = trend.iloc[i-1]
        
        if RsiMa.iloc[i-1] > longband.iloc[i-1] and RSIndex > longband.iloc[i-1]:
            longband.iloc[i] = max(prev_longband, newlongband_val)
        else:
            longband.iloc[i] = newlongband_val
        
        if RsiMa.iloc[i-1] < shortband.iloc[i-1] and RSIndex < shortband.iloc[i-1]:
            shortband.iloc[i] = min(prev_shortband, newshortband_val)
        else:
            shortband.iloc[i] = newshortband_val
        
        cross_1 = longband.iloc[i-1] < RSIndex and longband.iloc[i] >= RSIndex
        if RSIndex < shortband.iloc[i-1]:
            trend.iloc[i] = 1
        elif cross_1:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = prev_trend
        
        FastAtrRsiTL.iloc[i] = longband.iloc[i] if trend.iloc[i] == 1 else shortband.iloc[i]
    
    basis = (FastAtrRsiTL - 50).rolling(length_bb).mean()
    dev = qqeMult * (FastAtrRsiTL - 50).rolling(length_bb).std()
    upper = basis + dev
    lower = basis - dev
    
    # QQE 2
    RSI_Period2 = 6
    SF2 = 5
    QQE2 = 1.61
    ThreshHold2 = 3
    Wilders_Period2 = RSI_Period2 * 2 - 1
    
    Rsi2 = wilders_rsi(close, RSI_Period2)
    RsiMa2 = Rsi2.ewm(span=SF2, adjust=False).mean()
    AtrRsi2 = (RsiMa2.shift(1) - RsiMa2).abs()
    MaAtrRsi2 = AtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean()
    dar2 = MaAtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean() * QQE2
    
    longband2 = pd.Series(0.0, index=df.index)
    shortband2 = pd.Series(0.0, index=df.index)
    trend2 = pd.Series(1, index=df.index)
    
    for i in range(1, n):
        RSIndex2 = RsiMa2.iloc[i]
        DeltaFastAtrRsi2 = dar2.iloc[i]
        newshortband2_val = RSIndex2 + DeltaFastAtrRsi2
        newlongband2_val = RSIndex2 - DeltaFastAtrRsi2
        
        prev_longband2 = longband2.iloc[i-1]
        prev_shortband2 = shortband2.iloc[i-1]
        prev_trend2 = trend2.iloc[i-1]
        
        if RsiMa2.iloc[i-1] > longband2.iloc[i-1] and RSIndex2 > longband2.iloc[i-1]:
            longband2.iloc[i] = max(prev_longband2, newlongband2_val)
        else:
            longband2.iloc[i] = newlongband2_val
        
        if RsiMa2.iloc[i-1] < shortband2.iloc[i-1] and RSIndex2 < shortband2.iloc[i-1]:
            shortband2.iloc[i] = min(prev_shortband2, newshortband2_val)
        else:
            shortband2.iloc[i] = newshortband2_val
        
        cross_2 = longband2.iloc[i-1] < RSIndex2 and longband2.iloc[i] >= RSIndex2
        if RSIndex2 < shortband2.iloc[i-1]:
            trend2.iloc[i] = 1
        elif cross_2:
            trend2.iloc[i] = -1
        else:
            trend2.iloc[i] = prev_trend2
    
    Greenbar1 = RsiMa2 - 50 > ThreshHold2
    Greenbar2 = RsiMa - 50 > upper
    Redbar1 = RsiMa2 - 50 < -ThreshHold2
    Redbar2 = RsiMa - 50 < lower
    
    qqeBuy = (Greenbar1 & Greenbar2) if use_QQE else pd.Series(False, index=df.index)
    qqeSell = (Redbar1 & Redbar2) if use_QQE else pd.Series(False, index=df.index)
    
    # ========== SSL HYBRID ==========
    len_ssl = 60
    maType = 'HMA'
    
    def hma(source, length):
        wma1 = source.rolling(length // 2).mean() * 2
        wma2 = source.rolling(length).mean()
        return (wma1 - wma2).rolling(int(np.sqrt(length))).mean()
    
    sslBaseline = hma(close, len_ssl) if maType == 'HMA' else close.rolling(len_ssl).mean()
    sslUp = sslBaseline > sslBaseline.shift(1)
    sslDown = sslBaseline < sslBaseline.shift(1)
    
    sslBuy = sslUp if use_SSL else pd.Series(False, index=df.index)
    sslSell = sslDown if use_SSL else pd.Series(False, index=df.index)
    
    # ========== WADDAH ATTAR EXPLOSION ==========
    sensitivity = 150
    fastLength = 20
    slowLength = 40
    channelLength = 20
    mult_wae = 2.0
    
    def calc_macd_wae(source, fast, slow):
        fastMA = source.ewm(span=fast, adjust=False).mean()
        slowMA = source.ewm(span=slow, adjust=False).mean()
        return fastMA - slowMA
    
    def calc_bb_upper_wae(source, length, mult):
        basis = source.rolling(length).mean()
        dev = mult * source.rolling(length).std()
        return basis + dev
    
    def calc_bb_lower_wae(source, length, mult):
        basis = source.rolling(length).mean()
        dev = mult * source.rolling(length).std()
        return basis - dev
    
    macd_val = calc_macd_wae(close, fastLength, slowLength)
    t1 = (macd_val - macd_val.shift(1)) * sensitivity
    e1 = calc_bb_upper_wae(close, channelLength, mult_wae) - calc_bb_lower_wae(close, channelLength, mult_wae)
    
    trendUp_wae = t1 >= 0
    trendDown_wae = t1 < 0
    
    waeBuy = (trendUp_wae & (t1 > e1)) if use_WAE else pd.Series(False, index=df.index)
    waeSell = (trendDown_wae & (t1.abs() > e1)) if use_WAE else pd.Series(False, index=df.index)
    
    # ========== EHLERS TWO POLE SUPER SMOOTHER ==========
    PeriodE2PSS = 15
    inverseE2PSS = False
    
    pi_val = 2 * np.arcsin(1)
    a1_val = np.exp(-1.414 * pi_val / PeriodE2PSS)
    b1_val = 2 * a1_val * np.cos(1.414 * pi_val / PeriodE2PSS)
    coef2 = b1_val
    coef3 = -a1_val * a1_val
    coef1 = 1 - coef2 - coef3
    
    Filt2 = pd.Series(0.0, index=df.index)
    TriggerE2PSS = pd.Series(0.0, index=df.index)
    hl2 = (high + low) / 2
    
    for i in range(n):
        if i < 3:
            Filt2.iloc[i] = hl2.iloc[i]
        else:
            Filt2.iloc[i] = coef1 * hl2.iloc[i] + coef2 * Filt2.iloc[i-1] + coef3 * Filt2.iloc[i-2]
        TriggerE2PSS.iloc[i] = Filt2.iloc[i-1] if i > 0 else hl2.iloc[i]
    
    signalLongE2PSS = Filt2 > TriggerE2PSS
    signalShortE2PSS = Filt2 < TriggerE2PSS
    
    e2pssBuy = (signalLongE2PSS if not inverseE2PSS else signalShortE2PSS) if use_E2PSS else pd.Series(False, index=df.index)
    e2pssSell = (signalShortE2PSS if not inverseE2PSS else signalLongE2PSS) if use_E2PSS else pd.Series(False, index=df.index)
    
    # ========== TRENDILO/FLUXWAVE ==========
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    def alma(source, length, offset, sigma):
        m = offset * (length - 1)
        s = sigma * (length - 1) / 6
        w = np.exp(-np.power(np.arange(length) - m, 2) / (2 * s * s))
        w = w / w.sum()
        result = source.rolling(length).apply(lambda x: np.sum(w * x.values[::-1]), raw=True)
        return result
    
    pct_change = close.diff(trendilo_smooth) / close * 100
    avg_pct_change = alma(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    rms = trendilo_bmult * np.sqrt((avg_pct_change ** 2).rolling(trendilo_length).mean())
    
    trendilo_dir = pd.Series(0, index=df.index)
    trendilo_dir[avg_pct_change > rms] = 1
    trendilo_dir[avg_pct_change < -rms] = -1
    
    trendiloBuy = (trendilo_dir == 1) if use_Trendilo else pd.Series(False, index=df.index)
    trendiloSell = (trendilo_dir == -1) if use_Trendilo else pd.Series(False, index=df.index)
    
    # ========== TTM SQUEEZE ==========
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    KC_mult_low_TTMS = 2.0
    
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS_BB = BB_mult_TTMS * close.rolling(length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS_BB
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS_BB
    
    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    devKC_TTMS = tr.rolling(length_TTMS).mean()
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS
    
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    
    # ========== AGGREGATE SIGNALS ==========
    enabled_indicators = []
    long_conditions = []
    short_conditions = []
    
    if use_QQE:
        enabled_indicators.append('QQE')
        long_conditions.append(qqeBuy)
        short_conditions.append(qqeSell)
    
    if use_SSL:
        enabled_indicators.append('SSL')
        long_conditions.append(sslBuy)
        short_conditions.append(sslSell)
    
    if use_WAE:
        enabled_indicators.append('WAE')
        long_conditions.append(waeBuy)
        short_conditions.append(waeSell)
    
    if use_E2PSS:
        enabled_indicators.append('E2PSS')
        long_conditions.append(e2pssBuy)
        short_conditions.append(e2pssSell)
    
    if use_Trendilo:
        enabled_indicators.append('Trendilo')
        long_conditions.append(trendiloBuy)
        short_conditions.append(trendiloSell)
    
    if use_TTM:
        enabled_indicators.append('TTM')
        ttm_long = NoSqz_TTMS
        ttm_short = NoSqz_TTMS
        long_conditions.append(ttm_long)
        short_conditions.append(ttm_short)
    
    num_enabled = len(enabled_indicators)
    
    # Combine based on entry mode
    long_signal = pd.Series(False, index=df.index)
    short_signal = pd.Series(False, index=df.index)
    
    if num_enabled > 0:
        if entry_mode == "All Active":
            long_signal = long_conditions[0]
            short_signal = short_conditions[0]
            for i in range(1, num_enabled):
                long_signal = long_signal & long_conditions[i]
                short_signal = short_signal & short_conditions[i]
        elif entry_mode == "Majority Active":
            for i in range(num_enabled):
                if i == 0:
                    long_signal = long_conditions[i]
                    short_signal = short_conditions[i]
                else:
                    long_signal = long_signal | long_conditions[i]
                    short_signal = short_signal | short_conditions[i]
            majority_threshold = num_enabled // 2
            long_count = sum(long_conditions)
            short_count = sum(short_conditions)
            long_signal = long_count > majority_threshold
            short_signal = short_count > majority_threshold
        elif entry_mode == "Any Active":
            for i in range(num_enabled):
                if i == 0:
                    long_signal = long_conditions[i]
                    short_signal = short_conditions[i]
                else:
                    long_signal = long_signal | long_conditions[i]
                    short_signal = short_signal | short_conditions[i]
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(1, n):
        if np.isnan(qqeBuy.iloc[i]) or np.isnan(sslBuy.iloc[i]) or np.isnan(waeBuy.iloc[i]):
            continue
        
        if long_signal.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = float(close.iloc[i])
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
        elif short_signal.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = float(close.iloc[i])
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