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
    
    # ============ PARAMETERS (from Pine Script inputs) ============
    # QQE MOD
    RSI_PERIOD = 6
    SF = 6
    QQE_FACTOR = 3
    THRESHHOLD = 3
    RSI_PERIOD2 = 6
    SF2 = 5
    QQE2_FACTOR = 1.61
    THRESHHOLD2 = 3
    BB_LENGTH = 50
    BB_MULT = 0.35
    
    # SSL Hybrid
    SSL_LEN = 60
    SSL2_LEN = 5
    ATR_PERIOD = 14
    ATR_MULT = 1.0
    
    # Date range
    START_YEAR = 2022
    START_MONTH = 1
    START_DATE = 1
    END_YEAR = 2030
    END_MONTH = 1
    END_DATE = 1
    
    # ============ WILDER RSI ============
    def wilders_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi
    
    # ============ HMA (Hull Moving Average) ============
    def hma(src, length):
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        wma_half = src.rolling(window=half_length).apply(lambda x: np.dot(x, np.arange(1, half_length + 1)) / half_length, raw=True)
        wma_full = src.rolling(window=length).apply(lambda x: np.dot(x, np.arange(1, length + 1)) / length, raw=True)
        hma_val = 2 * wma_half - wma_full
        hma_result = hma_val.rolling(window=sqrt_length).apply(lambda x: np.dot(x, np.arange(1, sqrt_length + 1)) / sqrt_length, raw=True)
        return hma_result
    
    # ============ JMA (Jurik MA) simplified ============
    def jma(src, length, phase=3, power=1):
        alpha = 0.5 / (length + 1)
        beta = 0.5 / (length + 1)
        gamma = -alpha
        delta = 0.0
        jma = pd.Series(index=src.index, dtype=float)
        for i in range(len(src)):
            if i == 0:
                jma.iloc[i] = src.iloc[i]
            else:
                jma.iloc[i] = alpha * src.iloc[i] + (1 - alpha - beta) * jma.iloc[i-1]
        return jma
    
    # ============ QQE MOD CALCULATION ============
    Wilders_Period = RSI_PERIOD * 2 - 1
    
    Rsi = wilders_rsi(df['close'], RSI_PERIOD)
    RsiMa = Rsi.ewm(span=SF, adjust=False).mean()
    AtrRsi = np.abs(RsiMa.shift(1) - RsiMa)
    MaAtrRsi = AtrRsi.ewm(alpha=1.0/Wilders_Period, min_periods=Wilders_Period, adjust=False).mean()
    dar = MaAtrRsi.ewm(alpha=1.0/Wilders_Period, min_periods=Wilders_Period, adjust=False).mean() * QQE_FACTOR
    
    RSIndex = RsiMa
    
    longband = pd.Series(np.zeros(len(df)), index=df.index)
    shortband = pd.Series(np.zeros(len(df)), index=df.index)
    trend = pd.Series(np.zeros(len(df)), index=df.index)
    
    for i in range(1, len(df)):
        RSIndex_val = RSIndex.iloc[i]
        RSIndex_prev = RSIndex.iloc[i-1]
        longband_prev = longband.iloc[i-1]
        shortband_prev = shortband.iloc[i-1]
        
        newshortband = RSIndex_val + dar.iloc[i]
        newlongband = RSIndex_val - dar.iloc[i]
        
        if RSIndex_prev > longband_prev and RSIndex_val > longband_prev:
            longband.iloc[i] = max(longband_prev, newlongband)
        else:
            longband.iloc[i] = newlongband
        
        if RSIndex_prev < shortband_prev and RSIndex_val < shortband_prev:
            shortband.iloc[i] = min(shortband_prev, newshortband)
        else:
            shortband.iloc[i] = newshortband
        
        if RSIndex_val > shortband_prev:
            trend.iloc[i] = 1
        elif RSIndex_val < longband_prev:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
    
    FastAtrRsiTL = pd.Series(np.where(trend.values == 1, longband.values, shortband.values), index=df.index)
    
    # ============ QQE MOD 2 CALCULATION ============
    Wilders_Period2 = RSI_PERIOD2 * 2 - 1
    
    Rsi2 = wilders_rsi(df['close'], RSI_PERIOD2)
    RsiMa2 = Rsi2.ewm(span=SF2, adjust=False).mean()
    AtrRsi2 = np.abs(RsiMa2.shift(1) - RsiMa2)
    MaAtrRsi2 = AtrRsi2.ewm(alpha=1.0/Wilders_Period2, min_periods=Wilders_Period2, adjust=False).mean()
    dar2 = MaAtrRsi2.ewm(alpha=1.0/Wilders_Period2, min_periods=Wilders_Period2, adjust=False).mean() * QQE2_FACTOR
    
    RSIndex2 = RsiMa2
    
    longband2 = pd.Series(np.zeros(len(df)), index=df.index)
    shortband2 = pd.Series(np.zeros(len(df)), index=df.index)
    trend2 = pd.Series(np.zeros(len(df)), index=df.index)
    
    for i in range(1, len(df)):
        RSIndex2_val = RSIndex2.iloc[i]
        RSIndex2_prev = RSIndex2.iloc[i-1]
        longband2_prev = longband2.iloc[i-1]
        shortband2_prev = shortband2.iloc[i-1]
        
        newshortband2 = RSIndex2_val + dar2.iloc[i]
        newlongband2 = RSIndex2_val - dar2.iloc[i]
        
        if RSIndex2_prev > longband2_prev and RSIndex2_val > longband2_prev:
            longband2.iloc[i] = max(longband2_prev, newlongband2)
        else:
            longband2.iloc[i] = newlongband2
        
        if RSIndex2_prev < shortband2_prev and RSIndex2_val < shortband2_prev:
            shortband2.iloc[i] = min(shortband2_prev, newshortband2)
        else:
            shortband2.iloc[i] = newshortband2
        
        if RSIndex2_val > shortband2_prev:
            trend2.iloc[i] = 1
        elif RSIndex2_val < longband2_prev:
            trend2.iloc[i] = -1
        else:
            trend2.iloc[i] = trend2.iloc[i-1]
    
    FastAtrRsi2TL = pd.Series(np.where(trend2.values == 1, longband2.values, shortband2.values), index=df.index)
    
    # ============ BOLLINGER BANDS ON QQE ============
    QQE_BB_SRC = FastAtrRsiTL - 50
    basis = QQE_BB_SRC.rolling(window=BB_LENGTH).mean()
    std = QQE_BB_SRC.rolling(window=BB_LENGTH).std()
    upper = basis + BB_MULT * std
    lower = basis - BB_MULT * std
    
    # ============ QQE HISTOGRAM CONDITIONS ============
    Greenbar1 = (RsiMa2 - 50) > THRESHHOLD2
    Greenbar2 = (RsiMa - 50) > upper
    Redbar1 = (RsiMa2 - 50) < -THRESHHOLD2
    Redbar2 = (RsiMa - 50) < lower
    
    # ============ SSL HYBRID ============
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - df['close'].shift(1))
    tr3 = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_slen = tr.ewm(alpha=1.0/ATR_PERIOD, min_periods=ATR_PERIOD, adjust=False).mean()
    upper_band = df['close'] + atr_slen * ATR_MULT
    lower_band = df['close'] - atr_slen * ATR_MULT
    
    baseline = hma(df['close'], SSL_LEN)
    ssl2 = jma(df['close'], SSL2_LEN)
    
    # ============ DATE RANGE ============
    start_ts = pd.Timestamp(f'{START_YEAR}-{START_MONTH:02d}-{START_DATE:02d}', tz='UTC').timestamp()
    end_ts = pd.Timestamp(f'{END_YEAR}-{END_MONTH:02d}-{END_DATE:02d}', tz='UTC').timestamp()
    in_date_range = (df['time'] >= start_ts) & (df['time'] < end_ts)
    
    # ============ ENTRY CONDITIONS ============
    long_cond = Greenbar1 & Greenbar2 & (df['close'] > upper_band) & (df['close'] > baseline) & (df['close'] > ssl2)
    short_cond = Redbar1 & Redbar2 & (df['close'] < lower_band) & (df['close'] < baseline) & (df['close'] < ssl2)
    
    # ============ GENERATE ENTRIES ============
    entries = []
    trade_num = 0
    
    min_lookback = max(Wilders_Period, Wilders_Period2, BB_LENGTH, SSL_LEN, ATR_PERIOD) + 10
    
    for i in range(min_lookback, len(df)):
        if pd.isna(longband.iloc[i]) or pd.isna(baseline.iloc[i]) or pd.isna(upper.iloc[i]):
            continue
        
        if long_cond.iloc[i] and in_date_range.iloc[i]:
            trade_num += 1
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
        
        if short_cond.iloc[i] and in_date_range.iloc[i]:
            trade_num += 1
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
    
    return entries