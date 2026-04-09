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
    
    # ========== QQE MOD ==========
    RSI_Period = 6
    SF = 6
    QQE = 3
    Wilders_Period = RSI_Period * 2 - 1
    
    # Wilder's RSI
    def wilders_rsi(src, length):
        delta = src.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    Rsi = wilders_rsi(close, RSI_Period)
    RsiMa = Rsi.ewm(span=SF, adjust=False).mean()
    AtrRsi = (RsiMa.shift(1) - RsiMa).abs()
    MaAtrRsi = AtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean()
    dar = MaAtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean() * QQE
    
    RSIndex = RsiMa
    DeltaFastAtrRsi = dar
    
    longband = pd.Series(0.0, index=df.index)
    shortband = pd.Series(0.0, index=df.index)
    trend = pd.Series(0, index=df.index)
    
    for i in range(1, len(df)):
        newshortband = RSIndex.iloc[i] + DeltaFastAtrRsi.iloc[i]
        newlongband = RSIndex.iloc[i] - DeltaFastAtrRsi.iloc[i]
        
        if RSIndex.iloc[i-1] > longband.iloc[i-1] and RSIndex.iloc[i] > longband.iloc[i-1]:
            longband.iloc[i] = max(longband.iloc[i-1], newlongband)
        else:
            longband.iloc[i] = newlongband
            
        if RSIndex.iloc[i-1] < shortband.iloc[i-1] and RSIndex.iloc[i] < shortband.iloc[i-1]:
            shortband.iloc[i] = min(shortband.iloc[i-1], newshortband)
        else:
            shortband.iloc[i] = newshortband
            
        cross_1 = longband.iloc[i-1] < RSIndex.iloc[i-1] and longband.iloc[i] >= RSIndex.iloc[i]
        
        if RSIndex.iloc[i-1] < shortband.iloc[i-1] and RSIndex.iloc[i] >= shortband.iloc[i-1]:
            trend.iloc[i] = 1
        elif cross_1:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1] if i > 0 else 1
    
    FastAtrRsiTL = pd.Series(0.0, index=df.index)
    for i in range(len(df)):
        FastAtrRsiTL.iloc[i] = longband.iloc[i] if trend.iloc[i] == 1 else shortband.iloc[i]
    
    # Bollinger Bands on QQE
    length = 50
    qqeMult = 0.35
    basis = (FastAtrRsiTL - 50).rolling(length).mean()
    dev = (FastAtrRsiTL - 50).rolling(length).std(ddof=1) * qqeMult
    upper = basis + dev
    lower = basis - dev
    
    # Second QQE system
    RSI_Period2 = 6
    SF2 = 5
    QQE2 = 1.61
    Wilders_Period2 = RSI_Period2 * 2 - 1
    ThreshHold2 = 3
    
    Rsi2 = wilders_rsi(close, RSI_Period2)
    RsiMa2 = Rsi2.ewm(span=SF2, adjust=False).mean()
    AtrRsi2 = (RsiMa2.shift(1) - RsiMa2).abs()
    MaAtrRsi2 = AtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean()
    dar2 = MaAtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean() * QQE2
    
    RSIndex2 = RsiMa2
    DeltaFastAtrRsi2 = dar2
    
    longband2 = pd.Series(0.0, index=df.index)
    shortband2 = pd.Series(0.0, index=df.index)
    trend2 = pd.Series(0, index=df.index)
    
    for i in range(1, len(df)):
        newshortband2 = RSIndex2.iloc[i] + DeltaFastAtrRsi2.iloc[i]
        newlongband2 = RSIndex2.iloc[i] - DeltaFastAtrRsi2.iloc[i]
        
        if RSIndex2.iloc[i-1] > longband2.iloc[i-1] and RSIndex2.iloc[i] > longband2.iloc[i-1]:
            longband2.iloc[i] = max(longband2.iloc[i-1], newlongband2)
        else:
            longband2.iloc[i] = newlongband2
            
        if RSIndex2.iloc[i-1] < shortband2.iloc[i-1] and RSIndex2.iloc[i] < shortband2.iloc[i-1]:
            shortband2.iloc[i] = min(shortband2.iloc[i-1], newshortband2)
        else:
            shortband2.iloc[i] = newshortband2
            
        cross_2 = longband2.iloc[i-1] < RSIndex2.iloc[i-1] and longband2.iloc[i] >= RSIndex2.iloc[i]
        
        if RSIndex2.iloc[i-1] < shortband2.iloc[i-1] and RSIndex2.iloc[i] >= shortband2.iloc[i-1]:
            trend2.iloc[i] = 1
        elif cross_2:
            trend2.iloc[i] = -1
        else:
            trend2.iloc[i] = trend2.iloc[i-1] if i > 0 else 1
    
    FastAtrRsi2TL = pd.Series(0.0, index=df.index)
    for i in range(len(df)):
        FastAtrRsi2TL.iloc[i] = longband2.iloc[i] if trend2.iloc[i] == 1 else shortband2.iloc[i]
    
    # Entry conditions
    Greenbar1 = (RsiMa2 - 50) > ThreshHold2
    Greenbar2 = (RsiMa - 50) > upper
    Redbar1 = (RsiMa2 - 50) < -ThreshHold2
    Redbar2 = (RsiMa - 50) < lower
    
    long_condition = Greenbar1 & Greenbar2
    short_condition = Redbar1 & Redbar2
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i == 0:
            continue
        
        if pd.isna(long_condition.iloc[i]) or pd.isna(short_condition.iloc[i]):
            continue
        
        direction = None
        if long_condition.iloc[i]:
            direction = 'long'
        elif short_condition.iloc[i]:
            direction = 'short'
        
        if direction:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries