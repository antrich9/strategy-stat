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
    
    # ========== QQE MOD INDICATORS ==========
    RSI_Period = 6
    SF = 6
    QQE = 3
    Wilders_Period = RSI_Period * 2 - 1
    
    # RSI
    def calc_rsi(src, length):
        delta = src.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/Wilders_Period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/Wilders_Period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    Rsi = calc_rsi(df['close'], RSI_Period)
    RsiMa = Rsi.ewm(span=SF, adjust=False).mean()
    AtrRsi = abs(RsiMa.shift(1) - RsiMa)
    MaAtrRsi = AtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean()
    dar = MaAtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean() * QQE
    
    # QQE bands
    longband = pd.Series(0.0, index=df.index)
    shortband = pd.Series(0.0, index=df.index)
    trend = pd.Series(0, index=df.index)
    
    for i in range(1, len(df)):
        RSIndex = RsiMa.iloc[i]
        DeltaFastAtrRsi = dar.iloc[i]
        newshortband = RSIndex + DeltaFastAtrRsi
        newlongband = RSIndex - DeltaFastAtrRsi
        
        prev_longband = longband.iloc[i-1]
        prev_shortband = shortband.iloc[i-1]
        prev_RSIndex = RsiMa.iloc[i-1]
        
        if prev_RSIndex > prev_longband and RSIndex > prev_longband:
            longband.iloc[i] = max(prev_longband, newlongband)
        else:
            longband.iloc[i] = newlongband
            
        if prev_RSIndex < prev_shortband and RSIndex < prev_shortband:
            shortband.iloc[i] = min(prev_shortband, newshortband)
        else:
            shortband.iloc[i] = newshortband
        
        cross_1 = longband.iloc[i-1] < RSIndex and prev_RSIndex >= prev_longband
        
        if RSIndex < shortband.iloc[i-1]:
            trend.iloc[i] = 1
        elif cross_1:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1] if not pd.isna(trend.iloc[i-1]) else 1
    
    FastAtrRsiTL = pd.Series(np.where(trend == 1, longband, shortband), index=df.index)
    
    # QQE Bollinger
    length_bb = 50
    qqeMult = 0.35
    basis = (FastAtrRsiTL - 50).rolling(length_bb).mean()
    dev = qqeMult * (FastAtrRsiTL - 50).rolling(length_bb).std()
    upper = basis + dev
    lower = basis - dev
    
    # Zero cross counters
    QQEzlong = pd.Series(0, index=df.index)
    QQEzshort = pd.Series(0, index=df.index)
    
    for i in range(1, len(df)):
        if RsiMa.iloc[i] >= 50:
            QQEzlong.iloc[i] = QQEzlong.iloc[i-1] + 1
        else:
            QQEzlong.iloc[i] = 0
            
        if RsiMa.iloc[i] < 50:
            QQEzshort.iloc[i] = QQEzshort.iloc[i-1] + 1
        else:
            QQEzshort.iloc[i] = 0
    
    # QQE2
    RSI_Period2 = 6
    SF2 = 5
    QQE2 = 1.61
    Wilders_Period2 = RSI_Period2 * 2 - 1
    
    Rsi2 = calc_rsi(df['close'], RSI_Period2)
    RsiMa2 = Rsi2.ewm(span=SF2, adjust=False).mean()
    AtrRsi2 = abs(RsiMa2.shift(1) - RsiMa2)
    MaAtrRsi2 = AtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean()
    dar2 = MaAtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean() * QQE2
    
    longband2 = pd.Series(0.0, index=df.index)
    shortband2 = pd.Series(0.0, index=df.index)
    trend2 = pd.Series(0, index=df.index)
    
    for i in range(1, len(df)):
        RSIndex2 = RsiMa2.iloc[i]
        DeltaFastAtrRsi2 = dar2.iloc[i]
        newshortband2 = RSIndex2 + DeltaFastAtrRsi2
        newlongband2 = RSIndex2 - DeltaFastAtrRsi2
        
        prev_longband2 = longband2.iloc[i-1]
        prev_shortband2 = shortband2.iloc[i-1]
        prev_RSIndex2 = RsiMa2.iloc[i-1]
        
        if prev_RSIndex2 > prev_longband2 and RSIndex2 > prev_longband2:
            longband2.iloc[i] = max(prev_longband2, newlongband2)
        else:
            longband2.iloc[i] = newlongband2
            
        if prev_RSIndex2 < prev_shortband2 and RSIndex2 < prev_shortband2:
            shortband2.iloc[i] = min(prev_shortband2, newshortband2)
        else:
            shortband2.iloc[i] = newshortband2
        
        cross_2 = longband2.iloc[i-1] < RSIndex2 and prev_RSIndex2 >= prev_longband2
        
        if RSIndex2 < shortband2.iloc[i-1]:
            trend2.iloc[i] = 1
        elif cross_2:
            trend2.iloc[i] = -1
        else:
            trend2.iloc[i] = trend2.iloc[i-1] if not pd.isna(trend2.iloc[i-1]) else 1
    
    FastAtrRsi2TL = pd.Series(np.where(trend2 == 1, longband2, shortband2), index=df.index)
    
    # QQE2 zero cross
    QQE2zlong = pd.Series(0, index=df.index)
    QQE2zshort = pd.Series(0, index=df.index)
    
    for i in range(1, len(df)):
        if RsiMa2.iloc[i] >= 50:
            QQE2zlong.iloc[i] = QQE2zlong.iloc[i-1] + 1
        else:
            QQE2zlong.iloc[i] = 0
            
        if RsiMa2.iloc[i] < 50:
            QQE2zshort.iloc[i] = QQE2zshort.iloc[i-1] + 1
        else:
            QQE2zshort.iloc[i] = 0
    
    ThreshHold2 = 3
    
    Greenbar1 = RsiMa2 - 50 > ThreshHold2
    Greenbar2 = RsiMa - 50 > upper
    Redbar1 = RsiMa2 - 50 < -ThreshHold2
    Redbar2 = RsiMa - 50 < lower
    
    # ========== SSL HYBRID INDICATORS ==========
    len_ssl = 60
    
    # HMA function for SSL
    def hma(src, length):
        return pd.Series(
            2 * src.rolling(length // 2).apply(lambda x: x.ewm(span=length//2, adjust=False).mean().iloc[-1] if len(x) > 0 else np.nan, raw=False) 
            - src.ewm(span=length, adjust=False).mean(),
            index=src.index
        ).rolling(round(np.sqrt(length))).apply(lambda x: x.ewm(span=round(np.sqrt(length)), adjust=False).mean().iloc[-1] if len(x) > 0 else np.nan, raw=False)
    
    # Simple HMA approximation
    def calc_hma(src, length):
        half_len = length // 2
        sqrt_len = int(np.sqrt(length))
        wma_half = src.rolling(half_len).apply(lambda x: np.average(x, weights=range(1, half_len+1)) if len(x) >= half_len else np.nan, raw=True)
        wma_full = src.rolling(length).apply(lambda x: np.average(x, weights=range(1, length+1)) if len(x) >= length else np.nan, raw=True)
        hma_val = 2 * wma_half - wma_full
        hma_result = hma_val.rolling(sqrt_len).apply(lambda x: np.average(x, weights=range(1, sqrt_len+1)) if len(x) >= sqrt_len else np.nan, raw=True)
        return hma_result
    
    # JMA approximation (simplified)
    def calc_jma(src, length, phase=3, power=1):
        beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
        alpha = np.power(beta, power) * 2 * np.pi / (phase + 2)
        jma = pd.Series(0.0, index=src.index)
        for i in range(1, len(src)):
            if pd.isna(jma.iloc[i-1]):
                jma.iloc[i] = src.iloc[i]
            else:
                jma.iloc[i] = (1 - alpha) * jma.iloc[i-1] + alpha * src.iloc[i]
        return jma
    
    # Use HMA for SSL baseline
    ssl_baseline = calc_hma(df['close'], len_ssl)
    
    # SSL1 and SSL2 calculation (simplified)
    len2 = 5
    len3 = 15
    
    # SSL1 (fast) and SSL2 (slow) using HMA
    SSL1 = calc_hma(df['close'], len2)
    SSL2 = calc_hma(df['close'], len3)
    
    # SSL Hybrid signals
    ssl_long_cond = df['close'] > ssl_baseline
    ssl_short_cond = df['close'] < ssl_baseline
    
    # ========== WADDAH ATTAR EXPLOSION ==========
    # Simplified WAE implementation
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    
    # Explosion lines
    diff_ema = ema20 - ema50
    dead_cross = (diff_ema.shift(1) > 0) & (diff_ema < 0)
    golden_cross = (diff_ema.shift(1) < 0) & (diff_ema > 0)
    
    # Trend direction
    wa_long = ema20 > ema50
    wa_short = ema20 < ema50
    
    # ========== ENTRY CONDITIONS ==========
    # Long entry: QQE bullish + SSL bullish + WAE bullish
    long_cond = (Greenbar1 & Greenbar2) & ssl_long_cond & wa_long
    
    # Short entry: QQE bearish + SSL bearish + WAE bearish
    short_cond = (Redbar1 & Redbar2) & ssl_short_cond & wa_short
    
    # ========== GENERATE ENTRIES ==========
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(ssl_baseline.iloc[i]):
            continue
        
        if long_cond.iloc[i]:
            entry_price = df['close'].iloc[i]
            entry_ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(entry_ts),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
            
        elif short_cond.iloc[i]:
            entry_price = df['close'].iloc[i]
            entry_ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(entry_ts),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
    
    return entries