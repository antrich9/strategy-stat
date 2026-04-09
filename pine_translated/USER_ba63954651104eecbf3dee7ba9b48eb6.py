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
    
    # =============================================================================
    # QQE MOD INDICATORS
    # =============================================================================
    
    # RSI Periods and smoothing
    RSI_Period = 6
    SF = 6
    QQE = 3
    Wilders_Period = RSI_Period * 2 - 1
    
    # RSI calculation using Wilder's method
    def calc_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    Rsi = calc_rsi(close, RSI_Period)
    RsiMa = Rsi.ewm(span=SF, adjust=False).mean()
    AtrRsi = (RsiMa.shift(1) - RsiMa).abs()
    MaAtrRsi = AtrRsi.ewm(alpha=1.0/Wilders_Period, adjust=False).mean()
    dar = MaAtrRsi.ewm(alpha=1.0/Wilders_Period, adjust=False).mean() * QQE
    
    # QQE bands and trend
    longband = pd.Series(0.0, index=df.index)
    shortband = pd.Series(0.0, index=df.index)
    trend = pd.Series(0, index=df.index)
    RSIndex = RsiMa.copy()
    FastAtrRsiTL = pd.Series(0.0, index=df.index)
    
    for i in range(1, len(df)):
        DeltaFastAtrRsi = dar.iloc[i]
        RSIndex_val = RSIndex.iloc[i]
        newshortband_val = RSIndex_val + DeltaFastAtrRsi
        newlongband_val = RSIndex_val - DeltaFastAtrRsi
        
        if RSIndex.iloc[i-1] > longband.iloc[i-1] and RSIndex_val > longband.iloc[i-1]:
            longband.iloc[i] = max(longband.iloc[i-1], newlongband_val)
        else:
            longband.iloc[i] = newlongband_val
            
        if RSIndex.iloc[i-1] < shortband.iloc[i-1] and RSIndex_val < shortband.iloc[i-1]:
            shortband.iloc[i] = min(shortband.iloc[i-1], newshortband_val)
        else:
            shortband.iloc[i] = newshortband_val
            
        cross_long = longband.iloc[i-1] < RSIndex_val and RSIndex.iloc[i-1] >= longband.iloc[i-1]
        cross_short = RSIndex_val < shortband.iloc[i-1] and RSIndex.iloc[i-1] <= shortband.iloc[i-1]
        
        if cross_short:
            trend.iloc[i] = 1
        elif cross_long:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1] if not pd.isna(trend.iloc[i-1]) else 1
        
        FastAtrRsiTL.iloc[i] = longband.iloc[i] if trend.iloc[i] == 1 else shortband.iloc[i]
    
    # QQE2 settings
    RSI_Period2 = 6
    SF2 = 5
    QQE2 = 1.61
    Wilders_Period2 = RSI_Period2 * 2 - 1
    
    Rsi2 = calc_rsi(close, RSI_Period2)
    RsiMa2 = Rsi2.ewm(span=SF2, adjust=False).mean()
    AtrRsi2 = (RsiMa2.shift(1) - RsiMa2).abs()
    MaAtrRsi2 = AtrRsi2.ewm(alpha=1.0/Wilders_Period2, adjust=False).mean()
    dar2 = MaAtrRsi2.ewm(alpha=1.0/Wilders_Period2, adjust=False).mean() * QQE2
    
    longband2 = pd.Series(0.0, index=df.index)
    shortband2 = pd.Series(0.0, index=df.index)
    trend2 = pd.Series(0, index=df.index)
    RSIndex2 = RsiMa2.copy()
    FastAtrRsi2TL = pd.Series(0.0, index=df.index)
    
    for i in range(1, len(df)):
        DeltaFastAtrRsi2 = dar2.iloc[i]
        RSIndex2_val = RSIndex2.iloc[i]
        newshortband2_val = RSIndex2_val + DeltaFastAtrRsi2
        newlongband2_val = RSIndex2_val - DeltaFastAtrRsi2
        
        if RSIndex2.iloc[i-1] > longband2.iloc[i-1] and RSIndex2_val > longband2.iloc[i-1]:
            longband2.iloc[i] = max(longband2.iloc[i-1], newlongband2_val)
        else:
            longband2.iloc[i] = newlongband2_val
            
        if RSIndex2.iloc[i-1] < shortband2.iloc[i-1] and RSIndex2_val < shortband2.iloc[i-1]:
            shortband2.iloc[i] = min(shortband2.iloc[i-1], newshortband2_val)
        else:
            shortband2.iloc[i] = newshortband2_val
            
        cross_long2 = longband2.iloc[i-1] < RSIndex2_val and RSIndex2.iloc[i-1] >= longband2.iloc[i-1]
        cross_short2 = RSIndex2_val < shortband2.iloc[i-1] and RSIndex2.iloc[i-1] <= shortband2.iloc[i-1]
        
        if cross_short2:
            trend2.iloc[i] = 1
        elif cross_long2:
            trend2.iloc[i] = -1
        else:
            trend2.iloc[i] = trend2.iloc[i-1] if not pd.isna(trend2.iloc[i-1]) else 1
        
        FastAtrRsi2TL.iloc[i] = longband2.iloc[i] if trend2.iloc[i] == 1 else shortband2.iloc[i]
    
    # QQE Histogram bars
    ThreshHold = 3
    ThreshHold2 = 3
    
    # Bollinger bands on QQE
    length = 50
    qqeMult = 0.35
    basis = (FastAtrRsiTL - 50).rolling(length).mean()
    dev = qqeMult * (FastAtrRsiTL - 50).rolling(length).std()
    upper = basis + dev
    lower = basis - dev
    
    # Green/Red bars
    Greenbar1 = (RsiMa2 - 50) > ThreshHold2
    Greenbar2 = (RsiMa - 50) > upper
    Redbar1 = (RsiMa2 - 50) < -(ThreshHold2)
    Redbar2 = (RsiMa - 50) < lower
    
    # =============================================================================
    # SSL HYBRID INDICATORS
    # =============================================================================
    
    # ATR bands
    atrlen = 14
    mult = 1.0
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_slen = tr.ewm(alpha=1.0/atrlen, adjust=False).mean()
    upper_band = close + atr_slen * mult
    lower_band = close - atr_slen * mult
    
    # HMA function (Hull Moving Average)
    def hma(src, length):
        return 2 * src.rolling(int(length/2)).apply(lambda x: x.ewm(span=int(length/2), adjust=False).mean().iloc[-1], raw=False) - src.rolling(length).apply(lambda x: x.ewm(span=length, adjust=False).mean().iloc[-1], raw=False)
    
    # Simple EMA for baseline
    len_baseline = 60
    baseline = close.ewm(span=len_baseline, adjust=False).mean()
    
    # SSL1 and SSL2
    len_ssl1 = 60
    len_ssl2 = 5
    
    # Use HMA for SSL calculations
    ssl1 = hma(close, len_ssl1)
    ssl2 = hma(close, len_ssl2)
    
    # SSL Baseline and exit
    ssl_exit = hma(close, 15)
    
    # SSL Hybrid color (green if close > ssl, red if close < ssl)
    ssl_up = close > ssl1
    ssl_down = close < ssl1
    
    # =============================================================================
    # WADDAH ATTAR EXPLOSION
    # =============================================================================
    
    # Calculate momentum
    expl_len = 20
    deadzone = 20
    
    # Explosion line (EMA of price changes)
    price_change = close.diff()
    explosion = price_change.ewm(span=expl_len, adjust=False).mean() * 100
    dead = price_change.ewm(span=deadzone, adjust=False).mean() * 100
    
    # Trend direction based on explosion vs dead
    waddah_bullish = explosion > dead
    waddah_bearish = explosion < dead
    
    # =============================================================================
    # ENTRY CONDITIONS
    # =============================================================================
    
    # Long entry: QQE green bars + Waddah Attar bullish
    long_condition = (Greenbar1 & Greenbar2) & waddah_bullish
    
    # Short entry: QQE red bars + Waddah Attar bearish
    short_condition = (Redbar1 & Redbar2) & waddah_bearish
    
    # =============================================================================
    # GENERATE ENTRIES
    # =============================================================================
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        entry_ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        entry_price = float(df['close'].iloc[i])
        
        if long_condition.iloc[i] and not pd.isna(close.iloc[i]):
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_condition.iloc[i] and not pd.isna(close.iloc[i]):
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries