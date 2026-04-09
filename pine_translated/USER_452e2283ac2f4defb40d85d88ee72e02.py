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
    
    # Parameters
    # Ehlers Two Pole Super Smoother
    PeriodE2PSS = 15
    PriceE2PSS = (df['high'] + df['low']) / 2  # hl2
    
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * 3.14159 / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    # Ehlers Filter calculation
    Filt2 = np.zeros(len(df))
    Filt2[0] = PriceE2PSS.iloc[0]
    Filt2[1] = PriceE2PSS.iloc[1]
    
    for i in range(2, len(df)):
        Filt2[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
    
    TriggerE2PSS = np.roll(Filt2, 1)
    TriggerE2PSS[0] = np.nan
    
    signalLongE2PSS = Filt2 > TriggerE2PSS
    signalShortE2PSS = Filt2 < TriggerE2PSS
    
    # QQE Mod
    RSI_Period = 6
    SF = 6
    QQE = 3
    ThreshHold = 3
    
    Wilders_Period = RSI_Period * 2 - 1
    
    # RSI calculation
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/Wilders_Period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/Wilders_Period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    Rsi = 100 - (100 / (1 + rs))
    
    RsiMa = Rsi.ewm(span=SF, adjust=False).mean()
    
    AtrRsi = np.abs(RsiMa.shift(1) - RsiMa)
    MaAtrRsi = AtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean()
    dar = MaAtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean() * QQE
    
    # QQE Bands
    longband = np.zeros(len(df))
    shortband = np.zeros(len(df))
    trend = np.zeros(len(df))
    
    for i in range(1, len(df)):
        RSIndex = RsiMa.iloc[i]
        DeltaFastAtrRsi = dar.iloc[i]
        newshortband = RSIndex + DeltaFastAtrRsi
        newlongband = RSIndex - DeltaFastAtrRsi
        
        if RsiMa.iloc[i-1] > longband[i-1] and RSIndex > longband[i-1]:
            longband[i] = max(longband[i-1], newlongband)
        else:
            longband[i] = newlongband
        
        if RsiMa.iloc[i-1] < shortband[i-1] and RSIndex < shortband[i-1]:
            shortband[i] = min(shortband[i-1], newshortband)
        else:
            shortband[i] = newshortband
        
        if i > 0:
            cross_1 = longband[i-1] == RSIndex
            if RSIndex == shortband[i-1]:
                trend[i] = 1
            elif cross_1:
                trend[i] = -1
            else:
                trend[i] = trend[i-1] if not np.isnan(trend[i-1]) else 1
    
    FastAtrRsiTL = pd.Series(np.where(trend == 1, longband, shortband), index=df.index)
    
    # Bollinger on QQE
    length = 50
    qqeMult = 0.35
    basis = (FastAtrRsiTL - 50).rolling(length).mean()
    dev = qqeMult * (FastAtrRsiTL - 50).rolling(length).std()
    upper = basis + dev
    lower = basis - dev
    
    # QQE Mod #2
    RSI_Period2 = 6
    SF2 = 5
    QQE2 = 1.61
    ThreshHold2 = 3
    
    Wilders_Period2 = RSI_Period2 * 2 - 1
    
    Rsi2 = 100 - (100 / (1 + rs))  # Using same RSI calculation
    RsiMa2 = Rsi2.ewm(span=SF2, adjust=False).mean()
    
    AtrRsi2 = np.abs(RsiMa2.shift(1) - RsiMa2)
    MaAtrRsi2 = AtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean()
    dar2 = MaAtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean() * QQE2
    
    # QQE2 Bands
    longband2 = np.zeros(len(df))
    shortband2 = np.zeros(len(df))
    trend2 = np.zeros(len(df))
    
    for i in range(1, len(df)):
        RSIndex2 = RsiMa2.iloc[i]
        DeltaFastAtrRsi2 = dar2.iloc[i]
        newshortband2 = RSIndex2 + DeltaFastAtrRsi2
        newlongband2 = RSIndex2 - DeltaFastAtrRsi2
        
        if RsiMa2.iloc[i-1] > longband2[i-1] and RSIndex2 > longband2[i-1]:
            longband2[i] = max(longband2[i-1], newlongband2)
        else:
            longband2[i] = newlongband2
        
        if RsiMa2.iloc[i-1] < shortband2[i-1] and RSIndex2 < shortband2[i-1]:
            shortband2[i] = min(shortband2[i-1], newshortband2)
        else:
            shortband2[i] = newshortband2
        
        if i > 0:
            cross_2 = longband2[i-1] == RSIndex2
            if RSIndex2 == shortband2[i-1]:
                trend2[i] = 1
            elif cross_2:
                trend2[i] = -1
            else:
                trend2[i] = trend2[i-1] if not np.isnan(trend2[i-1]) else 1
    
    # QQE signals
    Greenbar1 = RsiMa2 - 50 > ThreshHold2
    Greenbar2 = RsiMa - 50 > upper
    Redbar1 = RsiMa2 - 50 < 0 - ThreshHold2
    Redbar2 = RsiMa - 50 < lower
    
    qqeBuy = Greenbar1 & Greenbar2
    qqeSell = Redbar1 & Redbar2
    
    # Trendilo
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    pct_change = df['close'].pct_change(trendilo_smooth) * 100
    
    # ALMA calculation
    def alma(src, length, offset, sigma):
        m = np.arange(length) + 1
        w = np.exp(-np.square(m - length * offset) / (2 * np.square(sigma)))
        w_sum = np.sum(w)
        w = w / w_sum
        return np.convolve(src, w, mode='valid')[:len(src)]
    
    avg_pct_change = pd.Series(alma(pct_change.values, trendilo_length, trendilo_offset, trendilo_sigma), index=df.index)
    rms = trendilo_bmult * np.sqrt((avg_pct_change * avg_pct_change).rolling(trendilo_length).sum() / trendilo_length)
    
    trendilo_dir = pd.Series(np.where(avg_pct_change > rms, 1, np.where(avg_pct_change < -rms, -1, 0)), index=df.index)
    
    trendiloBuy = trendilo_dir == 1
    trendiloSell = trendilo_dir == -1
    
    # TTM Squeeze
    use_TTMS = True
    highlightMovements_TTMS = True
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    KC_mult_low_TTMS = 2.0
    
    BB_basis_TTMS = df['close'].rolling(length_TTMS).mean()
    dev_TTMS = BB_mult_TTMS * df['close'].rolling(length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS
    
    KC_basis_TTMS = df['close'].rolling(length_TTMS).mean()
    devKC_TTMS = df['tr'].rolling(length_TTMS).mean()
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS
    
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    
    # Linear regression for momentum
    highest_high = df['high'].rolling(length_TTMS).max()
    lowest_low = df['low'].rolling(length_TTMS).min()
    mom_TTMS = (df['close'] - ((highest_high + lowest_low) / 2 + df['close'].rolling(length_TTMS).mean()) / 2).rolling(length_TTMS).mean()
    
    if highlightMovements_TTMS:
        ttmBuy = NoSqz_TTMS & (mom_TTMS > 0) & (mom_TTMS > mom_TTMS.shift(1))
        ttmSell = NoSqz_TTMS & (mom_TTMS < 0) & (mom_TTMS < mom_TTMS.shift(1))
    else:
        ttmBuy = mom_TTMS > 0
        ttmSell = mom_TTMS < 0
    
    # SSL Hybrid
    use_ssl = True
    ssl_len = 60
    
    def hma_calc(src, length):
        half_length = int(length / 2)
        sqrt_length = int(np.round(np.sqrt(length)))
        wma1 = 2 * src.rolling(half_length).mean() - src.rolling(length).mean()
        return wma1.rolling(sqrt_length).mean()
    
    ssl_baseline = hma_calc(df['close'], ssl_len)
    ssl_down = hma_calc(df['low'], ssl_len)
    ssl_up = hma_calc(df['high'], ssl_len)
    
    ssl_hlv = ssl_baseline.shift(1).isna() * 0 | (ssl_baseline > ssl_baseline.shift(1)) * 1 | (ssl_baseline < ssl_baseline.shift(1)) * -1
    
    sslBuy = ssl_hlv > 0
    sslSell = ssl_hlv < 0
    
    # Volume Filter
    use_vol_filter = True
    vol_length = 50
    vol_threshold = 75
    
    vol_ma = df['volume'].rolling(vol_length).mean()
    vol_std = df['volume'].rolling(vol_length).std()
    norm_vol = (df['volume'] - vol_ma) / vol_std * 100 + 100
    
    volFilter = (~use_vol_filter) | (norm_vol > vol_threshold)
    
    # Combined entry conditions
    longCondition = signalLongE2PSS & qqeBuy & trendiloBuy & ttmBuy & sslBuy & volFilter
    shortCondition = signalShortE2PSS & qqeSell & trendiloSell & ttmSell & sslSell & volFilter
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if longCondition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': df['time'].iloc[i],
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif shortCondition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': df['time'].iloc[i],
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries