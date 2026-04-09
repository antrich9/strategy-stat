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
    
    # Initialize result list
    entries = []
    trade_num = 1
    
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # ----- QQE MOD INDICATOR -----
    # First QQE
    RSI_Period = 6
    SF = 6
    QQE = 3
    ThreshHold = 3
    Wilders_Period = RSI_Period * 2 - 1
    
    # Calculate RSI
    def calculate_wilder_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    Rsi = calculate_wilder_rsi(data['close'], RSI_Period)
    RsiMa = Rsi.ewm(span=SF, adjust=False).mean()
    AtrRsi = np.abs(RsiMa.shift(1) - RsiMa)
    MaAtrRsi = AtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean()
    dar = MaAtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean() * QQE
    
    # QQE bands calculation
    RSIndex = RsiMa
    longband = pd.Series(0.0, index=data.index)
    shortband = pd.Series(0.0, index=data.index)
    trend = pd.Series(0, index=data.index)
    FastAtrRsiTL = pd.Series(0.0, index=data.index)
    
    DeltaFastAtrRsi = dar
    newshortband = RSIndex + DeltaFastAtrRsi
    newlongband = RSIndex - DeltaFastAtrRsi
    
    for i in range(1, len(data)):
        if RSIndex.iloc[i-1] > longband.iloc[i-1] and RSIndex.iloc[i] > longband.iloc[i]:
            longband.iloc[i] = max(longband.iloc[i-1], newlongband.iloc[i])
        else:
            longband.iloc[i] = newlongband.iloc[i]
            
        if RSIndex.iloc[i-1] < shortband.iloc[i-1] and RSIndex.iloc[i] < shortband.iloc[i]:
            shortband.iloc[i] = min(shortband.iloc[i-1], newshortband.iloc[i])
        else:
            shortband.iloc[i] = newshortband.iloc[i]
    
    # Trend calculation
    for i in range(1, len(data)):
        if RSIndex.iloc[i] > shortband.iloc[i-1]:
            trend.iloc[i] = 1
        elif longband.iloc[i-1] > RSIndex.iloc[i]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
    
    FastAtrRsiTL = pd.Series(0.0, index=data.index)
    for i in range(len(data)):
        if trend.iloc[i] == 1:
            FastAtrRsiTL.iloc[i] = longband.iloc[i]
        else:
            FastAtrRsiTL.iloc[i] = shortband.iloc[i]
    
    # Zero cross counters
    QQEzlong = pd.Series(0, index=data.index)
    QQEzshort = pd.Series(0, index=data.index)
    
    for i in range(1, len(data)):
        QQEzlong.iloc[i] = QQEzlong.iloc[i-1] + 1 if RSIndex.iloc[i] >= 50 else 0
        QQEzshort.iloc[i] = QQEzshort.iloc[i-1] + 1 if RSIndex.iloc[i] < 50 else 0
    
    # Second QQE
    RSI_Period2 = 6
    SF2 = 5
    QQE2 = 1.61
    ThreshHold2 = 3
    Wilders_Period2 = RSI_Period2 * 2 - 1
    
    Rsi2 = calculate_wilder_rsi(data['close'], RSI_Period2)
    RsiMa2 = Rsi2.ewm(span=SF2, adjust=False).mean()
    AtrRsi2 = np.abs(RsiMa2.shift(1) - RsiMa2)
    MaAtrRsi2 = AtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean()
    dar2 = MaAtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean() * QQE2
    
    RSIndex2 = RsiMa2
    longband2 = pd.Series(0.0, index=data.index)
    shortband2 = pd.Series(0.0, index=data.index)
    trend2 = pd.Series(0, index=data.index)
    FastAtrRsi2TL = pd.Series(0.0, index=data.index)
    
    DeltaFastAtrRsi2 = dar2
    newshortband2 = RSIndex2 + DeltaFastAtrRsi2
    newlongband2 = RSIndex2 - DeltaFastAtrRsi2
    
    for i in range(1, len(data)):
        if RSIndex2.iloc[i-1] > longband2.iloc[i-1] and RSIndex2.iloc[i] > longband2.iloc[i]:
            longband2.iloc[i] = max(longband2.iloc[i-1], newlongband2.iloc[i])
        else:
            longband2.iloc[i] = newlongband2.iloc[i]
            
        if RSIndex2.iloc[i-1] < shortband2.iloc[i-1] and RSIndex2.iloc[i] < shortband2.iloc[i]:
            shortband2.iloc[i] = min(shortband2.iloc[i-1], newshortband2.iloc[i])
        else:
            shortband2.iloc[i] = newshortband2.iloc[i]
    
    for i in range(1, len(data)):
        if RSIndex2.iloc[i] > shortband2.iloc[i-1]:
            trend2.iloc[i] = 1
        elif longband2.iloc[i-1] > RSIndex2.iloc[i]:
            trend2.iloc[i] = -1
        else:
            trend2.iloc[i] = trend2.iloc[i-1]
    
    for i in range(len(data)):
        if trend2.iloc[i] == 1:
            FastAtrRsi2TL.iloc[i] = longband2.iloc[i]
        else:
            FastAtrRsi2TL.iloc[i] = shortband2.iloc[i]
    
    QQE2zlong = pd.Series(0, index=data.index)
    QQE2zshort = pd.Series(0, index=data.index)
    
    for i in range(1, len(data)):
        QQE2zlong.iloc[i] = QQE2zlong.iloc[i-1] + 1 if RSIndex2.iloc[i] >= 50 else 0
        QQE2zshort.iloc[i] = QQE2zshort.iloc[i-1] + 1 if RSIndex2.iloc[i] < 50 else 0
    
    # QQE bars conditions
    Greenbar1 = RsiMa2 - 50 > ThreshHold2
    Greenbar2 = RsiMa - 50 > (pd.Series(0.0, index=data.index))  # Simplified upper threshold
    Redbar1 = RsiMa2 - 50 < 0 - ThreshHold2
    Redbar2 = RsiMa - 50 < -(pd.Series(0.0, index=data.index))  # Simplified lower threshold
    
    # ----- SSL HYBRID INDICATOR -----
    # ATR bands
    atrlen = 14
    mult = 1.0
    data['tr'] = np.maximum(data['high'] - data['low'],
                           np.maximum(np.abs(data['high'] - data['close'].shift(1)),
                                     np.abs(data['low'] - data['close'].shift(1))))
    atr_slen = data['tr'].ewm(alpha=1/atrlen, adjust=False).mean()
    upper_band = atr_slen * mult + data['close']
    lower_band = data['close'] - atr_slen * mult
    
    # Baseline (HMA by default)
    len_baseline = 60
    
    def hma(src, length):
        wma1 = src.rolling(length // 2).mean() * 2
        wma2 = src.rolling(length).mean()
        return (wma1 - wma2).rolling(int(np.sqrt(length))).mean()
    
    baseline = hma(data['close'], len_baseline)
    
    # SSL1
    len_ssl1 = 5
    ssl1 = hma(data['close'], len_ssl1)
    
    # SSL2
    len_ssl2 = 5
    ssl2 = hma(data['close'], len_ssl2)
    
    # SSL Baseline crossover conditions
    ssl_long_cond = (ssl1 > baseline) & (ssl1.shift(1) <= baseline.shift(1))
    ssl_short_cond = (ssl1 < baseline) & (ssl1.shift(1) >= baseline.shift(1))
    
    # ----- WADDAH ATTAR EXPLOSION -----
    ema20 = data['close'].ewm(span=20, adjust=False).mean()
    ema50 = data['close'].ewm(span=50, adjust=False).mean()
    
    # Calculate MACD for WAE
    ema12 = data['close'].ewm(span=12, adjust=False).mean()
    ema26 = data['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    
    # Explosion calculation
    sensitivity = 150
    multiplier = 20
    
    diff = ema20 - ema50
    dead_line = -diff.rolling(15).std() * multiplier
    weave_line = diff.rolling(15).std() * multiplier
    
    explosion_long = (diff > weave_line) & (diff.shift(1) <= weave_line.shift(1))
    explosion_short = (diff < dead_line) & (diff.shift(1) >= dead_line.shift(1))
    
    # ----- ENTRY CONDITIONS -----
    # Long entry: QQE shows strength + SSL bullish + optional WAE confirmation
    long_condition = (Greenbar1 & Greenbar2) & (ssl1 > baseline) & (RSIndex2 > 50)
    
    # Short entry: QQE shows weakness + SSL bearish + optional WAE confirmation
    short_condition = (Redbar1 & Redbar2) & (ssl1 < baseline) & (RSIndex2 < 50)
    
    # Alternative QQE crossover signals
    qqe_long_cross = (RSIndex2 > 50) & (RSIndex2.shift(1) <= 50)
    qqe_short_cross = (RSIndex2 < 50) & (RSIndex2.shift(1) >= 50)
    
    # Combined long condition with QQE crossover
    final_long = (qqe_long_cross) & (ssl1 > baseline) & (RsiMa2 > ThreshHold2)
    
    # Combined short condition with QQE crossover
    final_short = (qqe_short_cross) & (ssl1 < baseline) & (RsiMa2 < -ThreshHold2)
    
    # Iterate through bars and generate entries
    for i in range(len(data)):
        if pd.isna(longband.iloc[i]) or pd.isna(RsiMa.iloc[i]) or pd.isna(ssl1.iloc[i]) or pd.isna(baseline.iloc[i]):
            continue
            
        if final_long.iloc[i]:
            entry_price = data['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(data['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(data['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
            
        elif final_short.iloc[i]:
            entry_price = data['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(data['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(data['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
    
    return entries