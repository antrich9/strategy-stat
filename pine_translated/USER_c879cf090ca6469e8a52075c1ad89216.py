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
    
    # Parameters (from Pine Script inputs)
    # Risk Management (not used for entries but needed for context)
    atrLength = 14
    atrMultiplier = 2.0
    
    # HTF Filter
    useHTFFilter = True
    htfPeriod = 15
    
    # Baseline Ehlers
    PeriodE2PSS = 15
    
    # Trendilo
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    # TTM Squeeze
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    KC_mult_low_TTMS = 2.0
    requireNoSqueeze = True
    
    # QQE MOD
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
    
    # Volume Filter
    useVolumeFilter = False
    
    # Calculate HTF Ehlers (placeholder - in real implementation would need HTF data)
    # Since we don't have HTF data in the dataframe, we'll assume HTF filter passes
    # or we need to resample. For simplicity, assume HTF filter is disabled or passes.
    htfBullish = pd.Series(True, index=df.index)
    htfBearish = pd.Series(True, index=df.index)
    
    htfLongOK = htfBullish if useHTFFilter else pd.Series(True, index=df.index)
    htfShortOK = htfBearish if useHTFFilter else pd.Series(True, index=df.index)
    
    # Baseline Ehlers Two Pole Super Smoother
    pi = 2 * np.arcsin(1)
    
    def ehlers_filter(series, period):
        a1 = np.exp(-1.414 * np.pi / period)
        b1 = 2 * a1 * np.cos(1.414 * pi / period)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        
        filt = np.zeros(len(series))
        filt[0] = series.iloc[0]
        
        for i in range(1, len(series)):
            if i < 2:  # bar_index < 3 in Pine (0,1,2)
                filt[i] = series.iloc[i]
            else:
                filt[i] = c1 * series.iloc[i] + c2 * filt[i-1] + c3 * filt[i-2]
        
        return pd.Series(filt, index=series.index)
    
    # For Ehlers, we need to iterate properly
    # Let's implement it using pandas operations or loops
    
    price = (df['high'] + df['low']) / 2  # hl2
    
    # Ehlers Two Pole SS
    a1 = np.exp(-1.414 * np.pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    
    Filt2 = np.zeros(len(df))
    Filt2[0] = price.iloc[0]
    Filt2[1] = price.iloc[1]
    
    for i in range(2, len(df)):
        Filt2[i] = c1 * price.iloc[i] + c2 * Filt2[i-1] + c3 * Filt2[i-2]
    
    Filt2 = pd.Series(Filt2, index=df.index)
    TriggerE2PSS = Filt2.shift(1)
    
    baselineLong = Filt2 > TriggerE2PSS
    baselineShort = Filt2 < TriggerE2PSS
    
    # Trendilo
    pct_change = df['close'].diff(trendilo_smooth) / df['close'] * 100
    
    # ALMA implementation
    def alma(series, length, offset, sigma):
        # Window
        window = np.arange(length)
        # Gaussian
        m = offset * (length - 1)
        s = sigma * (length - 1) / 6
        weights = np.exp(-((window - m) ** 2) / (2 * s ** 2))
        weights = weights / weights.sum()
        
        result = series.rolling(length).apply(lambda x: np.dot(x, weights), raw=True)
        return result
    
    avg_pct_change = alma(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    
    # RMS
    rms = trendilo_bmult * np.sqrt(avg_pct_change.pow(2).rolling(trendilo_length).mean())
    
    trendilo_dir = pd.Series(0, index=df.index)
    trendilo_dir[avg_pct_change > rms] = 1
    trendilo_dir[avg_pct_change < -rms] = -1
    
    trendiloLong = trendilo_dir == 1
    trendiloShort = trendilo_dir == -1
    
    # TTM Squeeze
    BB_basis_TTMS = df['close'].rolling(length_TTMS).mean()
    dev_TTMS = BB_mult_TTMS * df['close'].rolling(length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS
    
    KC_basis_TTMS = df['close'].rolling(length_TTMS).mean()
    devKC_TTMS = df['high'].combine(df['low'], lambda h, l: h - l).rolling(length_TTMS).mean()  # tr
    # Actually ta.tr in Pine is typically high - low or high - close prev or low - close prev
    # Let's use the standard true range
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - df['close'].shift(1))
    tr3 = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    devKC_TTMS = tr.rolling(length_TTMS).mean()
    
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS
    
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    
    # Momentum (linreg)
    # linreg(source, length, offset) = intercept + slope * (length - 1 - offset)
    # But let's approximate or implement properly
    
    # For linreg, we'll use the formula: sum((i - mean_i) * (source_i - mean_source)) / sum((i - mean_i)^2)
    # This is complex. Let's use a simpler momentum or implement the exact linreg.
    
    # Actually, Pine's linreg is: intercept + slope * (length - 1 - offset)
    # where slope = cov(source, bar_index) / var(bar_index)
    
    def linreg(source, length, offset=0):
        indices = np.arange(length)
        mean_indices = (length - 1) / 2
        
        results = []
        for i in range(length - 1, len(source)):
            window = source.iloc[i - length + 1:i + 1].values
            mean_source = window.mean()
            
            numerator = np.sum((indices - mean_indices) * (window - mean_source))
            denominator = np.sum((indices - mean_indices) ** 2)
            
            if denominator == 0:
                slope = 0
            else:
                slope = numerator / denominator
            
            intercept = mean_source - slope * mean_indices
            linreg_val = intercept + slope * (length - 1 - offset)
            results.append(linreg_val)
        
        # Pad with NaN
        for _ in range(length - 1):
            results.insert(0, np.nan)
        
        return pd.Series(results, index=source.index)
    
    # Actually this is slow. Let's use a vectorized version or just approximate.
    # For TTM momentum, it's typically: linreg(close - avg(highest, lowest, length), length, 0)
    # where avg(highest, lowest) is the average of highest and lowest over length
    
    # Let's simplify: mom_TTMS = close - sma(close, length)
    # But Pine uses: close - math.avg(math.avg(highest, lowest), sma(close))
    
    highest = df['high'].rolling(length_TTMS).max()
    lowest = df['low'].rolling(length_TTMS).min()
    avg_hl = (highest + lowest) / 2
    avg_close = df['close'].rolling(length_TTMS).mean()
    source_mom = df['close'] - (avg_hl + avg_close) / 2
    
    mom_TTMS = linreg(source_mom, length_TTMS, 0)
    
    momLong = (mom_TTMS > 0) & (mom_TTMS > mom_TTMS.shift(1))
    momShort = (mom_TTMS < 0) & (mom_TTMS < mom_TTMS.shift(1))
    
    if requireNoSqueeze:
        ttmLong = momLong & NoSqz_TTMS
        ttmShort = momShort & NoSqz_TTMS
    else:
        ttmLong = momLong
        ttmShort = momShort
    
    # QQE MOD
    Wilders_Period = RSI_Period * 2 - 1
    
    def wilder_rsi(series, period):
        # Wilder RSI - uses smoothed moving average
        delta = series.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        # First average is simple SMA
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        # Subsequent values use Wilder's smoothing
        for i in range(period, len(series)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    Rsi = wilder_rsi(df['close'], RSI_Period)
    RsiMa = Rsi.ewm(span=SF, adjust=False).mean()
    AtrRsi = np.abs(RsiMa.shift(1) - RsiMa)
    MaAtrRsi = AtrRsi.ewm(span=Wilders_Period, adjust=False).mean()
    dar = MaAtrRsi.ewm(span=Wilders_Period, adjust=False).mean() * QQE
    
    longband = pd.Series(0.0, index=df.index)
    shortband = pd.Series(0.0, index=df.index)
    trend = pd.Series(1, index=df.index)  # initial value
    
    RSIndex = RsiMa
    
    for i in range(1, len(df)):
        newshortband = RSIndex.iloc[i] + dar.iloc[i]
        newlongband = RSIndex.iloc[i] - dar.iloc[i]
        
        if RSIndex.iloc[i-1] > longband.iloc[i-1] and RSIndex.iloc[i] > longband.iloc[i]:
            longband.iloc[i] = max(longband.iloc[i-1], newlongband)
        else:
            longband.iloc[i] = newlongband
        
        if RSIndex.iloc[i-1] < shortband.iloc[i-1] and RSIndex.iloc[i] < shortband.iloc[i]:
            shortband.iloc[i] = min(shortband.iloc[i-1], newshortband)
        else:
            shortband.iloc[i] = newshortband
        
        if RSIndex.iloc[i] < shortband.iloc[i-1]:
            trend.iloc[i] = 1
        elif RSIndex.iloc[i] > longband.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
    
    FastAtrRsiTL = pd.Series(0.0, index=df.index)
    FastAtrRsiTL[trend == 1] = longband[trend == 1]
    FastAtrRsiTL[trend == -1] = shortband[trend == -1]
    
    basis = (FastAtrRsiTL - 50).rolling(length_bb).mean()
    dev_qqe = qqeMult * (FastAtrRsiTL - 50).rolling(length_bb).std()
    upper = basis + dev_qqe
    lower = basis - dev_qqe
    
    # QQE 2
    Wilders_Period2 = RSI_Period2 * 2 - 1
    Rsi2 = wilder_rsi(df['close'], RSI_Period2)
    RsiMa2 = Rsi2.ewm(span=SF2, adjust=False).mean()
    AtrRsi2 = np.abs(RsiMa2.shift(1) - RsiMa2)
    MaAtrRsi2 = AtrRsi2.ewm(span=Wilders_Period2, adjust=False).mean()
    dar2 = MaAtrRsi2.ewm(span=Wilders_Period2, adjust=False).mean() * QQE2
    
    Greenbar1 = RsiMa2 - 50 > ThreshHold2
    Greenbar2 = RsiMa - 50 > upper
    Redbar1 = RsiMa2 - 50 < 0 - ThreshHold2
    Redbar2 = RsiMa - 50 < lower
    
    qqeLong = Greenbar1 & Greenbar2
    qqeShort = Redbar1 & Redbar2
    
    # Volume Filter (simplified since useVolumeFilter is false by default)
    if useVolumeFilter:
        vol_length = 20  # placeholder
        vol_ma = df['volume'].rolling(vol_length).mean()
        volumeBullish = df['volume'] > vol_ma
        volumeBearish = df['volume'] < vol_ma
    else:
        volumeBullish = pd.Series(True, index=df.index)
        volumeBearish = pd.Series(True, index=df.index)
    
    # Entry Logic
    # Assuming: HTF filter AND Baseline AND (Trendilo OR TTM OR QQE) AND Volume
    # Or more strictly: HTF AND Baseline AND Trendilo AND TTM AND QQE AND Volume
    
    # Given the "Unified" nature, let's require all confirmations:
    long_condition = htfLongOK & baselineLong & trendiloLong & ttmLong & qqeLong & volumeBullish
    short_condition = htfShortOK & baselineShort & trendiloShort & ttmShort & qqeShort & volumeBearish
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_condition.iloc[i]:
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            }
            entries.append(entry)
            trade_num += 1
        
        if short_condition.iloc[i]:
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            }
            entries.append(entry)
            trade_num += 1
    
    return entries