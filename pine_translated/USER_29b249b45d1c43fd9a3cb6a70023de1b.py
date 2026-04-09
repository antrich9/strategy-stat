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
    
    use_QQE = True
    use_SSL = True
    use_WAE = True
    use_E2PSS = True
    use_Trendilo = True
    
    entry_mode = "All Active"
    
    RSI_Period = 6
    SF = 6
    QQE = 3
    ThreshHold = 3
    Wilders_Period = RSI_Period * 2 - 1
    qqeMult = 0.35
    length = 50
    
    Rsi = close.rolling(RSI_Period).apply(lambda x: 100 - 100 / (1 + (x.diff().clip(lower=0).rolling(RSI_Period).sum() / (-x.diff().clip(upper=0)).rolling(RSI_Period).sum()).iloc[-1]) if (-x.diff().clip(upper=0)).rolling(RSI_Period).sum().iloc[-1] != 0 else 50, raw=False)
    Rsi = close.rolling(RSI_Period).apply(lambda x: 100 - 100 / (1 + x.diff().clip(lower=0).sum() / (-x.diff().clip(upper=0)).sum()) if x.diff().clip(upper=0).sum() != 0 else 50, raw=False)
    
    def wilders_rsi(series, period):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    Rsi = wilders_rsi(close, RSI_Period)
    RsiMa = Rsi.ewm(span=SF, adjust=False).mean()
    AtrRsi = (RsiMa.shift(1) - RsiMa).abs()
    MaAtrRsi = AtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean()
    dar = MaAtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean() * QQE
    
    longband = pd.Series(np.zeros(len(close)), index=close.index)
    shortband = pd.Series(np.zeros(len(close)), index=close.index)
    trend = pd.Series(np.zeros(len(close)), index=close.index)
    
    for i in range(1, len(close)):
        RSIndex = RsiMa.iloc[i]
        DeltaFastAtrRsi = dar.iloc[i]
        newshortband_val = RSIndex + DeltaFastAtrRsi
        newlongband_val = RSIndex - DeltaFastAtrRsi
        
        if RsiMa.iloc[i-1] > longband.iloc[i-1] and RSIndex > longband.iloc[i-1]:
            longband.iloc[i] = max(longband.iloc[i-1], newlongband_val)
        else:
            longband.iloc[i] = newlongband_val
        
        if RsiMa.iloc[i-1] < shortband.iloc[i-1] and RSIndex < shortband.iloc[i-1]:
            shortband.iloc[i] = min(shortband.iloc[i-1], newshortband_val)
        else:
            shortband.iloc[i] = newshortband_val
        
        if RSIndex > shortband.iloc[i-1]:
            trend.iloc[i] = 1
        elif RSIndex < longband.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
    
    FastAtrRsiTL = pd.Series(np.where(trend == 1, longband.values, shortband.values), index=close.index)
    FastAtrRsiTL_minus50 = FastAtrRsiTL - 50
    basis = FastAtrRsiTL_minus50.rolling(length).mean()
    dev = qqeMult * FastAtrRsiTL_minus50.rolling(length).std(ddof=0)
    upper = basis + dev
    lower = basis - dev
    
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
    
    longband2 = pd.Series(np.zeros(len(close)), index=close.index)
    shortband2 = pd.Series(np.zeros(len(close)), index=close.index)
    trend2 = pd.Series(np.zeros(len(close)), index=close.index)
    
    for i in range(1, len(close)):
        RSIndex2 = RsiMa2.iloc[i]
        DeltaFastAtrRsi2 = dar2.iloc[i]
        newshortband2_val = RSIndex2 + DeltaFastAtrRsi2
        newlongband2_val = RSIndex2 - DeltaFastAtrRsi2
        
        if RsiMa2.iloc[i-1] > longband2.iloc[i-1] and RSIndex2 > longband2.iloc[i-1]:
            longband2.iloc[i] = max(longband2.iloc[i-1], newlongband2_val)
        else:
            longband2.iloc[i] = newlongband2_val
        
        if RsiMa2.iloc[i-1] < shortband2.iloc[i-1] and RSIndex2 < shortband2.iloc[i-1]:
            shortband2.iloc[i] = min(shortband2.iloc[i-1], newshortband2_val)
        else:
            shortband2.iloc[i] = newshortband2_val
        
        if RSIndex2 > shortband2.iloc[i-1]:
            trend2.iloc[i] = 1
        elif RSIndex2 < longband2.iloc[i-1]:
            trend2.iloc[i] = -1
        else:
            trend2.iloc[i] = trend2.iloc[i-1]
    
    Greenbar1 = RsiMa2 - 50 > ThreshHold2
    Greenbar2 = RsiMa - 50 > upper
    Redbar1 = RsiMa2 - 50 < -ThreshHold2
    Redbar2 = RsiMa - 50 < lower
    
    qqeBuy = Greenbar1 & Greenbar2 if use_QQE else pd.Series(False, index=close.index)
    qqeSell = Redbar1 & Redbar2 if use_QQE else pd.Series(False, index=close.index)
    
    maType = 'HMA'
    len_ssl = 60
    
    def hma(source, length):
        return 2 * source.ewm(span=length//2, adjust=False).mean() - source.ewm(span=length, adjust=False).mean()
    
    def ma_function(source, length, ma_type):
        if ma_type == 'SMA':
            return source.rolling(length).mean()
        elif ma_type == 'EMA':
            return source.ewm(span=length, adjust=False).mean()
        elif ma_type == 'WMA':
            weights = np.arange(1, length + 1)
            return source.rolling(length).apply(lambda x: np.dot(x, weights[:len(x)]) / weights[:len(x)].sum(), raw=True)
        elif ma_type == 'HMA':
            return hma(source, length)
        else:
            return source.rolling(length).mean()
    
    sslBaseline = ma_function(close, len_ssl, maType)
    sslUp = sslBaseline > sslBaseline.shift(1)
    sslDown = sslBaseline < sslBaseline.shift(1)
    
    sslBuy = sslUp if use_SSL else pd.Series(False, index=close.index)
    sslSell = sslDown if use_SSL else pd.Series(False, index=close.index)
    
    sensitivity = 150
    fastLength = 20
    slowLength = 40
    channelLength = 20
    mult_wae = 2.0
    
    fastMA_wae = close.ewm(span=fastLength, adjust=False).mean()
    slowMA_wae = close.ewm(span=slowLength, adjust=False).mean()
    macd_val = fastMA_wae - slowMA_wae
    macd_prev = macd_val.shift(1)
    
    t1 = (macd_val - macd_prev) * sensitivity
    e1_basis = close.rolling(channelLength).mean()
    e1_dev = mult_wae * close.rolling(channelLength).std(ddof=0)
    e1 = (e1_basis + e1_dev) - (e1_basis - e1_dev)
    
    trendUp_wae = t1 >= 0
    trendDown_wae = t1 < 0
    
    waeBuy = (trendUp_wae & (t1 > e1)) if use_WAE else pd.Series(False, index=close.index)
    waeSell = (trendDown_wae & (t1.abs() > e1)) if use_WAE else pd.Series(False, index=close.index)
    
    PeriodE2PSS = 15
    inverseE2PSS = False
    PriceE2PSS = (high + low) / 2
    
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    filt2 = np.zeros(len(close))
    filt2[0] = PriceE2PSS.iloc[0]
    filt2[1] = PriceE2PSS.iloc[1]
    
    for i in range(2, len(close)):
        filt2[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * filt2[i-1] + coef3 * filt2[i-2]
    
    Filt2 = pd.Series(filt2, index=close.index)
    TriggerE2PSS = Filt2.shift(1)
    TriggerE2PSS.iloc[0] = TriggerE2PSS.iloc[1]
    
    signalLongE2PSS = Filt2 > TriggerE2PSS
    signalShortE2PSS = Filt2 < TriggerE2PSS
    
    e2