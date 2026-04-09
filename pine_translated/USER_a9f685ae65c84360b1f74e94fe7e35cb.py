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
    
    close = df['close'].copy()
    lengthT3 = 5
    factor = 0.7
    srcT3 = close
    srcTrendilo = close
    smooth = 1
    lengthTrendilo = 50
    offset = 0.85
    sigma = 6
    bmult = 1.0
    blen = 20
    
    # Wilder RSI implementation
    def wilder_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Wilder ATR implementation
    def wilder_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    # ALMA implementation
    def alma(series, length, offset_val, sigma_val):
        window = np.arange(length)
        m = offset_val * (length - 1)
        s = length / sigma_val if sigma_val > 0 else length / 6
        weights = np.exp(-((window - m) ** 2) / (2 * s * s))
        weights = weights / weights.sum()
        
        result = pd.Series(index=series.index, dtype=float)
        for i in range(length - 1, len(series)):
            result.iloc[i] = np.sum(weights * series.iloc[i - length + 1:i + 1].values)
        return result
    
    # T3 calculation
    def calc_gd(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factor) - ema2 * factor
    
    t3 = calc_gd(calc_gd(calc_gd(srcT3, lengthT3), lengthT3), lengthT3)
    
    # Trendilo calculation
    pch = close.diff(smooth) / close.shift(smooth) * 100
    avpch = alma(pch, lengthTrendilo, offset, sigma)
    
    # rms calculation
    blength = blen  # cblen is false
    rolling_sum = avpch.rolling(window=blength).apply(lambda x: (x * x).sum(), raw=True)
    rms = bmult * np.sqrt(rolling_sum / blength)
    
    cdir = pd.Series(index=close.index, dtype=int)
    cdir = np.where(avpch > rms, 1, np.where(avpch < -rms, -1, 0))
    cdir = pd.Series(cdir, index=close.index)
    
    # RSI
    rsi_value = wilder_rsi(close, 14)
    rsi_overbought = rsi_value > 70
    rsi_oversold = rsi_value < 30
    
    # Entry conditions
    longCondition = (close > t3) & (cdir == 1) & (~rsi_overbought)
    shortCondition = (close < t3) & (cdir == -1) & (~rsi_oversold)
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i == 0:
            continue
        
        if pd.isna(t3.iloc[i]) or pd.isna(avpch.iloc[i]) or pd.isna(rms.iloc[i]) or pd.isna(rsi_value.iloc[i]):
            continue
        
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = float(close.iloc[i])
        
        if longCondition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if shortCondition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
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