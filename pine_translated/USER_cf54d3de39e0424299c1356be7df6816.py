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
    
    # Inputs
    PriceE2PSS = (df['high'] + df['low']) / 2
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
    
    RSI_Period2 = 6
    SF2 = 5
    QQE2 = 1.61
    ThreshHold2 = 3
    
    length_bb = 50
    qqeMult = 0.35
    
    useVolumeFilter = False
    vol_length = 50
    vol_threshold = 75
    
    requireAllConfirmations = True
    
    # Ehlers Two Pole Super Smoother
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * np.pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    Filt2 = np.zeros(len(df))
    TriggerE2PSS = np.zeros(len(df))
    Filt2[0] = PriceE2PSS.iloc[0]
    Filt2[1] = PriceE2PSS.iloc[1]
    TriggerE2PSS[0] = PriceE2PSS.iloc[0]
    
    for i in range(2, len(df)):
        Filt2[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
    
    TriggerE2PSS[1:] = Filt2[:-1]
    
    baselineLong = pd.Series(Filt2 > TriggerE2PSS, index=df.index)
    baselineShort = pd.Series(Filt2 < TriggerE2PSS, index=df.index)
    
    # Trendilo
    pct_change = close.pct_change(trendilo_smooth) * 100
    
    def alma(arr, length, offset, sigma):
        w = np.arange(length)
        w = np.exp(-((w - offset * (length - 1)) ** 2) / (2 * sigma ** 2 * length))
        w = w / w.sum()
        return pd.Series(np.convolve(arr, w, mode='same'), index=arr.index)
    
    avg_pct_change = alma(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    rms_arr = np.sqrt((avg_pct_change ** 2).rolling(trendilo_length).mean())
    rms = trendilo_bmult * rms_arr
    
    trendilo_dir = pd.Series(0, index=df.index)
    trendilo_dir[avg_pct_change > rms] = 1
    trendilo_dir[avg_pct_change < -rms] = -1
    
    trendiloLong = trendilo_dir == 1
    trendiloShort = trendilo_dir == -1
    
    # TTM Squeeze
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = close.rolling(length_TTMS).std() * BB_mult_TTMS
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS
    
    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    tr = df['high'].combine_max(df['close'].shift(1)) - df['low'].combine_min(df['close'].shift(1))
    devKC_TTMS = tr.rolling(length_TTMS).mean()
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS
    
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    
    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    avg_close = close.rolling(length_TTMS).mean()
    mom_TTMS = (close - (highest_high + lowest_low) / 2 - avg_close) * np.ones(length_TTMS)
    
    # Manual linreg approximation
    mom_TTMS = pd.Series(0.0, index=df.index)
    for i in range(length_TTMS - 1, len(df)):
        x = np.arange(length_TTMS)
        y = (close.iloc[i-length_TTMS+1:i+1] - (highest_high.iloc[i-length_TTMS+1:i+1] + lowest_low.iloc[i-length_TTMS+1:i+1]) / 2 - avg_close.iloc[i-length_TTMS+1:i+1]).values
        if len(y) == length_TTMS:
            y_mean = np.mean(y)
            num = np.sum((x - (length_TTMS - 1) / 2) * (y - y_mean))
            denom = np.sum((x - (length_TTMS - 1) / 2) ** 2)
            if denom != 0:
                slope = num / denom
                intercept = y_mean - slope * (length_TTMS - 1) / 2
                mom_TTMS.iloc[i] = slope * (length_TTMS - 1) + intercept
    
    momLong = (mom_TTMS > 0) & (mom_TTMS > mom_TTMS.shift(1))
    momShort = (mom_TTMS < 0) & (mom_TTMS < mom_TTMS.shift(1))
    
    ttmLong = momLong & (NoSqz_TTMS if requireNoSqueeze else pd.Series(True, index=df.index))
    ttmShort = momShort & (NoSqz_TTMS if requireNoSqueeze else pd.Series(True, index=df.index))
    
    # QQE MOD
    Wilders_Period = RSI_Period * 2 - 1
    
    def wilder_rsi(src, length):
        delta = src.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    Rsi = wilder_rsi(close, RSI_Period)
    RsiMa = Rsi.ewm(span=SF, adjust=False).mean()
    AtrRsi = np.abs(RsiMa.shift(1) - RsiMa)
    MaAtrRsi = AtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean()
    dar = MaAtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean() * QQE
    
    longband = pd.Series(0.0, index=df.index)
    shortband = pd.Series(0.0, index=df.index)
    trend = pd.Series(0, index=df.index)
    
    RSIndex = RsiMa
    newshortband = RSIndex + dar
    newlongband = RSIndex - dar
    
    for i in range(1, len(df)):
        longband.iloc[i] = newlongband.iloc[i] if not (RSIndex.iloc[i-1] > longband.iloc[i-1] and RSIndex.iloc[i] > longband.iloc[i-1]) else max(longband.iloc[i-1], newlongband.iloc[i])
        shortband.iloc[i] = newshortband.iloc[i] if not (RSIndex.iloc[i-1] < shortband.iloc[i-1] and RSIndex.iloc[i] < shortband.iloc[i-1]) else min(shortband.iloc[i-1], newshortband.iloc[i])
        
        if RSIndex.iloc[i] < shortband.iloc[i-1] if i > 0 else False:
            trend.iloc[i] = 1
        elif RSIndex.iloc[i-1] > longband.iloc[i-1] if i > 0 else False:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1] if i > 0 else 1
    
    FastAtrRsiTL = pd.Series(0.0, index=df.index)
    FastAtrRsiTL[trend == 1] = longband[trend == 1]
    FastAtrRsiTL[trend == -1] = shortband[trend == -1]
    
    basis = (FastAtrRsiTL - 50).rolling(length_bb).mean()
    dev_qqe = (FastAtrRsiTL - 50).rolling(length_bb).std() * qqeMult
    upper = basis + dev_qqe
    lower = basis - dev_qqe
    
    # QQE2
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
    
    for i in range(1, len(df)):
        longband2.iloc[i] = newlongband2.iloc[i] if not (RSIndex2.iloc[i-1] > longband2.iloc[i-1] and RSIndex2.iloc[i] > longband2.iloc[i-1]) else max(longband2.iloc[i-1], newlongband2.iloc[i])
        shortband2.iloc[i] = newshortband2.iloc[i] if not (RSIndex2.iloc[i-1] < shortband2.iloc[i-1] and RSIndex2.iloc[i] < shortband2.iloc[i-1]) else min(shortband2.iloc[i-1], newshortband2.iloc[i])
    
    Greenbar1 = RsiMa2 - 50 > ThreshHold2
    Greenbar2 = RsiMa - 50 > upper
    Redbar1 = RsiMa2 - 50 < -ThreshHold2
    Redbar2 = RsiMa - 50 < lower
    
    qqeLong = Greenbar1 & Greenbar2
    qqeShort = Redbar1 & Redbar2
    
    # Volume Filter
    vol_avg = volume.rolling(vol_length).mean()
    vol_normalized = volume / vol_avg * 100
    volumeOK = vol_normalized > vol_threshold if useVolumeFilter else pd.Series(True, index=df.index)
    
    # Combined Entry Logic
    longConfirmations = (trendiloLong.astype(int)) + (ttmLong.astype(int)) + (qqeLong.astype(int))
    shortConfirmations = (trendiloShort.astype(int)) + (ttmShort.astype(int)) + (qqeShort.astype(int))
    
    if requireAllConfirmations:
        long_condition = baselineLong & trendiloLong & ttmLong & qqeLong & volumeOK
        short_condition = baselineShort & trendiloShort & ttmShort & qqeShort & volumeOK
    else:
        long_condition = baselineLong & (longConfirmations >= 2) & volumeOK
        short_condition = baselineShort & (shortConfirmations >= 2) & volumeOK
    
    # Time window check (London afternoon: 14:45 to 16:45)
    times = pd.to_datetime(df['time'], unit='s', utc=True)
    hour = times.dt.hour
    minute = times.dt.minute
    isWithinAfternoonWindow = (hour * 60 + minute >= 14 * 60 + 45) & (hour * 60 + minute < 16 * 60 + 45)
    in_trading_window = isWithinAfternoonWindow
    
    # Final conditions
    long_entry = long_condition & in_trading_window
    short_entry = short_condition & in_trading_window
    
    # Generate entries
    entries = []
    trade_num = 1
    
    long_triggered = False
    short_triggered = False
    
    for i in range(len(df)):
        if long_entry.iloc[i] and not long_triggered:
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
            long_triggered = True
        
        if short_entry.iloc[i] and not short_triggered:
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
            short_triggered = True
        
        # Reset trigger flags when condition becomes false
        if not long_entry.iloc[i]:
            long_triggered = False
        if not short_entry.iloc[i]:
            short_triggered = False
    
    return entries