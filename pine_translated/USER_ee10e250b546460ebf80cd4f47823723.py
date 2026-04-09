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
    open_price = df['open']
    
    # Strategy parameters
    use_QQE = True
    use_SSL = True
    use_WAE = True
    use_E2PSS = True
    use_Trendilo = True
    use_TTM = True
    use_ALMA = False
    use_QuickSilver = False
    
    # Indicator parameters
    RSI_Period = 6
    SF = 6
    QQE = 3
    ThreshHold = 3
    RSI_Period2 = 6
    SF2 = 5
    QQE2 = 1.61
    ThreshHold2 = 3
    qqeMult = 0.35
    length = 50
    
    maType = 'HMA'
    len_ssl = 60
    
    sensitivity = 150
    fastLength = 20
    slowLength = 40
    channelLength = 20
    mult_wae = 2.0
    
    PeriodE2PSS = 15
    inverseE2PSS = False
    
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    KC_mult_low_TTMS = 2.0
    
    # ===== Wilder's RSI =====
    def wilders_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # ===== Wilder's ATR =====
    def wilders_atr(high, low, close, period):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    # ===== Indicator 1: QQE Mod =====
    Wilders_Period = RSI_Period * 2 - 1
    Rsi = wilders_rsi(close, RSI_Period)
    RsiMa = Rsi.ewm(span=SF, adjust=False).mean()
    AtrRsi = abs(RsiMa.shift(1) - RsiMa)
    MaAtrRsi = AtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean()
    dar = MaAtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean() * QQE
    
    longband = pd.Series(0.0, index=close.index)
    shortband = pd.Series(0.0, index=close.index)
    trend = pd.Series(0, index=close.index)
    
    for i in range(len(close)):
        if i == 0:
            continue
        RSIndex = RsiMa.iloc[i]
        DeltaFastAtrRsi = dar.iloc[i]
        newshortband = RSIndex + DeltaFastAtrRsi
        newlongband = RSIndex - DeltaFastAtrRsi
        
        if RsiMa.iloc[i-1] > longband.iloc[i-1] and RSIndex > longband.iloc[i-1]:
            longband.iloc[i] = max(longband.iloc[i-1], newlongband)
        else:
            longband.iloc[i] = newlongband
            
        if RsiMa.iloc[i-1] < shortband.iloc[i-1] and RSIndex < shortband.iloc[i-1]:
            shortband.iloc[i] = min(shortband.iloc[i-1], newshortband)
        else:
            shortband.iloc[i] = newshortband
            
        if RSIndex < shortband.iloc[i-1]:
            trend.iloc[i] = 1
        elif RSIndex > longband.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
    
    FastAtrRsiTL = pd.Series(0.0, index=close.index)
    for i in range(len(close)):
        if trend.iloc[i] == 1:
            FastAtrRsiTL.iloc[i] = longband.iloc[i]
        else:
            FastAtrRsiTL.iloc[i] = shortband.iloc[i]
    
    basis = (FastAtrRsiTL - 50).rolling(length).mean()
    dev_qqe = qqeMult * (FastAtrRsiTL - 50).rolling(length).std()
    upper = basis + dev_qqe
    lower = basis - dev_qqe
    
    # QQE 2
    Wilders_Period2 = RSI_Period2 * 2 - 1
    Rsi2 = wilders_rsi(close, RSI_Period2)
    RsiMa2 = Rsi2.ewm(span=SF2, adjust=False).mean()
    AtrRsi2 = abs(RsiMa2.shift(1) - RsiMa2)
    MaAtrRsi2 = AtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean()
    dar2 = MaAtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean() * QQE2
    
    longband2 = pd.Series(0.0, index=close.index)
    shortband2 = pd.Series(0.0, index=close.index)
    trend2 = pd.Series(0, index=close.index)
    
    for i in range(len(close)):
        if i == 0:
            continue
        RSIndex2 = RsiMa2.iloc[i]
        DeltaFastAtrRsi2 = dar2.iloc[i]
        newshortband2 = RSIndex2 + DeltaFastAtrRsi2
        newlongband2 = RSIndex2 - DeltaFastAtrRsi2
        
        if RsiMa2.iloc[i-1] > longband2.iloc[i-1] and RSIndex2 > longband2.iloc[i-1]:
            longband2.iloc[i] = max(longband2.iloc[i-1], newlongband2)
        else:
            longband2.iloc[i] = newlongband2
            
        if RsiMa2.iloc[i-1] < shortband2.iloc[i-1] and RSIndex2 < shortband2.iloc[i-1]:
            shortband2.iloc[i] = min(shortband2.iloc[i-1], newshortband2)
        else:
            shortband2.iloc[i] = newshortband2
            
        if RSIndex2 < shortband2.iloc[i-1]:
            trend2.iloc[i] = 1
        elif RSIndex2 > longband2.iloc[i-1]:
            trend2.iloc[i] = -1
        else:
            trend2.iloc[i] = trend2.iloc[i-1]
    
    Greenbar1 = RsiMa2 - 50 > ThreshHold2
    Greenbar2 = RsiMa - 50 > upper
    Redbar1 = RsiMa2 - 50 < -ThreshHold2
    Redbar2 = RsiMa - 50 < lower
    
    qqeBuy = (Greenbar1 & Greenbar2) if use_QQE else pd.Series(False, index=close.index)
    qqeSell = (Redbar1 & Redbar2) if use_QQE else pd.Series(False, index=close.index)
    
    # ===== Indicator 2: SSL Hybrid =====
    def hma(src, length):
        return 2 * src.ewm(span=length//2, adjust=False).mean() - src.ewm(span=length, adjust=False).mean()
    
    def sma(src, length):
        return src.rolling(length).mean()
    
    def wma(src, length):
        weights = np.arange(1, length + 1)
        return src.rolling(length).apply(lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(), raw=True)
    
    def ma_function(source, length, ma_type):
        if ma_type == 'SMA':
            return sma(source, length)
        elif ma_type == 'EMA':
            return source.ewm(span=length, adjust=False).mean()
        elif ma_type == 'WMA':
            return wma(source, length)
        elif ma_type == 'HMA':
            return hma(source, length)
        else:
            return sma(source, length)
    
    sslBaseline = ma_function(close, len_ssl, maType)
    sslUp = sslBaseline > sslBaseline.shift(1)
    sslDown = sslBaseline < sslBaseline.shift(1)
    
    sslBuy = sslUp if use_SSL else pd.Series(False, index=close.index)
    sslSell = sslDown if use_SSL else pd.Series(False, index=close.index)
    
    # ===== Indicator 3: Waddah Attar Explosion =====
    def calc_macd(source, fastLength, slowLength):
        fastMA = source.ewm(span=fastLength, adjust=False).mean()
        slowMA = source.ewm(span=slowLength, adjust=False).mean()
        return fastMA - slowMA
    
    def calc_BBUpper(source, length, mult):
        basis = source.rolling(length).mean()
        dev = mult * source.rolling(length).std()
        return basis + dev
    
    def calc_BBLower(source, length, mult):
        basis = source.rolling(length).mean()
        dev = mult * source.rolling(length).std()
        return basis - dev
    
    t1 = (calc_macd(close, fastLength, slowLength) - calc_macd(close.shift(1), fastLength, slowLength)) * sensitivity
    e1 = calc_BBUpper(close, channelLength, mult_wae) - calc_BBLower(close, channelLength, mult_wae)
    
    trendUp = t1 >= 0
    trendDown = t1 < 0
    
    waeBuy = ((trendUp) & (t1 > e1)) if use_WAE else pd.Series(False, index=close.index)
    waeSell = ((trendDown) & (abs(t1) > e1)) if use_WAE else pd.Series(False, index=close.index)
    
    # ===== Indicator 4: Ehlers Two Pole Super Smoother =====
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    Filt2 = pd.Series(0.0, index=close.index)
    TriggerE2PSS = pd.Series(0.0, index=close.index)
    
    PriceE2PSS = (high + low) / 2
    
    for i in range(len(close)):
        if i < 2:
            Filt2.iloc[i] = PriceE2PSS.iloc[i]
        else:
            Filt2.iloc[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * Filt2.iloc[i-1] + coef3 * Filt2.iloc[i-2]
        TriggerE2PSS.iloc[i] = Filt2.iloc[i-1] if i > 0 else 0
    
    signalLongE2PSS = (Filt2 > TriggerE2PSS) if use_E2PSS else pd.Series(False, index=close.index)
    signalShortE2PSS = (Filt2 < TriggerE2PSS) if use_E2PSS else pd.Series(False, index=close.index)
    
    e2pssBuy = signalShortE2PSS if inverseE2PSS else signalLongE2PSS
    e2pssSell = signalLongE2PSS if inverseE2PSS else signalShortE2PSS
    
    # ===== Indicator 5: Trendilo/FluxWave =====
    def alma(src, length, offset, sigma):
        m = (offset * (length - 1))
        s = (length - 1) / 3.0
        w = np.exp(-((np.arange(length) - m) ** 2) / (2 * s ** 2))
        return src.rolling(length).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)
    
    pct_change = (close.diff(trendilo_smooth) / close) * 100
    avg_pct_change = alma(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    rms = trendilo_bmult * np.sqrt((avg_pct_change ** 2).rolling(trendilo_length).mean())
    
    trendilo_dir = pd.Series(0, index=close.index)
    trendilo_dir[avg_pct_change > rms] = 1
    trendilo_dir[avg_pct_change < -rms] = -1
    
    trendiloBuy = (trendilo_dir == 1) if use_Trendilo else pd.Series(False, index=close.index)
    trendiloSell = (trendilo_dir == -1) if use_Trendilo else pd.Series(False, index=close.index)
    
    # ===== Indicator 6: TTM Squeeze =====
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = BB_mult_TTMS * close.rolling(length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS
    
    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    devKC_TTMS = df['high'].diff().rolling(length_TTMS).mean()
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS
    
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    
    # Linear regression for momentum
    avg_high = high.rolling(length_TTMS).max()
    avg_low = low.rolling(length_TTMS).min()
    avg_close = close.rolling(length_TTMS).mean()
    mom_src = close - (avg_high + avg_low) / 2 - avg_close
    mom_TTMS = mom_src.rolling(length_TTMS).mean()  # Simplified linear regression
    
    ttmBuy = NoSqz_TTMS if use_TTM else pd.Series(False, index=close.index)
    ttmSell = NoSqz_TTMS if use_TTM else pd.Series(False, index=close.index)
    
    # ===== Combine Entry Signals =====
    active_indicators_buy = sum([use_QQE, use_SSL, use_WAE, use_E2PSS, use_Trendilo, use_TTM])
    active_indicators_sell = active_indicators_buy
    
    entry_mode = "All Active"
    
    if entry_mode == "All Active":
        long_condition = qqeBuy & sslBuy & waeBuy & e2pssBuy & trendiloBuy & ttmBuy
        short_condition = qqeSell & sslSell & waeSell & e2pssSell & trendiloSell & ttmSell
    elif entry_mode == "Majority Active":
        buy_votes = qqeBuy.astype(int) + sslBuy.astype(int) + waeBuy.astype(int) + e2pssBuy.astype(int) + trendiloBuy.astype(int) + ttmBuy.astype(int)
        sell_votes = qqeSell.astype(int) + sslSell.astype(int) + waeSell.astype(int) + e2pssSell.astype(int) + trendiloSell.astype(int) + ttmSell.astype(int)
        majority_threshold = active_indicators_buy / 2
        long_condition = buy_votes > majority_threshold
        short_condition = sell_votes > majority_threshold
    else:  # "Any Active"
        long_condition = qqeBuy | sslBuy | waeBuy | e2pssBuy | trendiloBuy | ttmBuy
        short_condition = qqeSell | sslSell | waeSell | e2pssSell | trendiloSell | ttmSell
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if np.isnan(RsiMa.iloc[i]) or np.isnan(sslBaseline.iloc[i]) or np.isnan(Filt2.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
        
        if short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
    
    return entries