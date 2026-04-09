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
    # Input parameters (matching Pine Script defaults)
    RSI_Period = 6
    SF = 6
    QQE = 3
    ThreshHold = 3
    RSI_Period2 = 6
    SF2 = 5
    QQE2 = 1.61
    ThreshHold2 = 3
    length = 50
    qqeMult = 0.35
    
    # Date range (matching Pine Script defaults)
    start_year, start_month, start_date = 2022, 1, 1
    end_year, end_month, end_date = 2030, 1, 1
    
    Wilders_Period = RSI_Period * 2 - 1
    Wilders_Period2 = RSI_Period2 * 2 - 1
    
    close = df['close'].copy()
    open_prices = df['open'].copy()
    high = df['high'].copy()
    low = df['low'].copy()
    
    # --- QQE MOD Calculations ---
    # RSI 1
    def wilder_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    Rsi = wilder_rsi(close, RSI_Period)
    RsiMa = Rsi.ewm(span=SF, adjust=False).mean()
    AtrRsi = (RsiMa.shift(1) - RsiMa).abs()
    MaAtrRsi = AtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean()
    dar = MaAtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean() * QQE
    
    RSIndex = RsiMa.copy()
    DeltaFastAtrRsi = dar.copy()
    newshortband = RSIndex + DeltaFastAtrRsi
    newlongband = RSIndex - DeltaFastAtrRsi
    
    longband = pd.Series(0.0, index=df.index)
    shortband = pd.Series(0.0, index=df.index)
    FastAtrRsiTL = pd.Series(0.0, index=df.index)
    
    for i in range(1, len(df)):
        prev_lb = longband.iloc[i-1] if not pd.isna(longband.iloc[i-1]) else 0.0
        prev_sb = shortband.iloc[i-1] if not pd.isna(shortband.iloc[i-1]) else 0.0
        
        new_long = newlongband.iloc[i]
        new_short = newshortband.iloc[i]
        
        if RSIndex.iloc[i-1] > prev_lb and RSIndex.iloc[i] > prev_lb:
            longband.iloc[i] = max(prev_lb, new_long)
        else:
            longband.iloc[i] = new_long
            
        if RSIndex.iloc[i-1] < prev_sb and RSIndex.iloc[i] < prev_sb:
            shortband.iloc[i] = min(prev_sb, new_short)
        else:
            shortband.iloc[i] = new_short
    
    FastAtrRsiTL = pd.Series(np.where(RSIndex >= 50, longband, shortband), index=df.index)
    
    basis = (FastAtrRsiTL - 50).rolling(length).mean()
    dev = qqeMult * (FastAtrRsiTL - 50).rolling(length).std()
    upper = basis + dev
    lower = basis - dev
    
    # RSI 2
    Rsi2 = wilder_rsi(close, RSI_Period2)
    RsiMa2 = Rsi2.ewm(span=SF2, adjust=False).mean()
    AtrRsi2 = (RsiMa2.shift(1) - RsiMa2).abs()
    MaAtrRsi2 = AtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean()
    dar2 = MaAtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean() * QQE2
    
    RSIndex2 = RsiMa2.copy()
    DeltaFastAtrRsi2 = dar2.copy()
    newshortband2 = RSIndex2 + DeltaFastAtrRsi2
    newlongband2 = RSIndex2 - DeltaFastAtrRsi2
    
    longband2 = pd.Series(0.0, index=df.index)
    shortband2 = pd.Series(0.0, index=df.index)
    FastAtrRsi2TL = pd.Series(0.0, index=df.index)
    
    for i in range(1, len(df)):
        prev_lb2 = longband2.iloc[i-1] if not pd.isna(longband2.iloc[i-1]) else 0.0
        prev_sb2 = shortband2.iloc[i-1] if not pd.isna(shortband2.iloc[i-1]) else 0.0
        
        new_long2 = newlongband2.iloc[i]
        new_short2 = newshortband2.iloc[i]
        
        if RSIndex2.iloc[i-1] > prev_lb2 and RSIndex2.iloc[i] > prev_lb2:
            longband2.iloc[i] = max(prev_lb2, new_long2)
        else:
            longband2.iloc[i] = new_long2
            
        if RSIndex2.iloc[i-1] < prev_sb2 and RSIndex2.iloc[i] < prev_sb2:
            shortband2.iloc[i] = min(prev_sb2, new_short2)
        else:
            shortband2.iloc[i] = new_short2
    
    FastAtrRsi2TL = pd.Series(np.where(RSIndex2 >= 50, longband2, shortband2), index=df.index)
    
    # --- QQE Entry Conditions ---
    Greenbar1 = (RsiMa2 - 50) > ThreshHold2
    Greenbar2 = (RsiMa - 50) > upper
    Redbar1 = (RsiMa2 - 50) < -ThreshHold2
    Redbar2 = (RsiMa - 50) < lower
    
    qqe_long_cond = Greenbar1 & Greenbar2
    qqe_short_cond = Redbar1 & Redbar2
    
    # --- HTF ALMA Filter (simplified - using Daily as default) ---
    # For a proper implementation, we'd need HTF data. Here we use current timeframe
    # and apply a simple ALMA approximation
    htf_alma_window = 9
    htf_alma_offset = 0.85
    htf_alma_sigma = 6
    
    def alma_approx(series, window, offset, sigma):
        m = (offset * (window - 1))
        s = sigma * (window - 1) / 6
        k = np.arange(window)
        w = np.exp(-np.power(k - m, 2) / (2 * s * s))
        w_sum = np.nansum(w)
        if w_sum == 0:
            return series.rolling(window).mean()
        w_norm = w / w_sum
        result = series.rolling(window).apply(lambda x: np.nansum(x * w_norm[::-1]), raw=True)
        return result
    
    htf_alma = alma_approx(close, htf_alma_window, htf_alma_offset, htf_alma_sigma)
    htf_bull = close > htf_alma
    htf_bear = close < htf_alma
    
    # --- Date Range Filter ---
    timestamps = pd.to_datetime(df['time'], unit='s', utc=True)
    start_dt = datetime(start_year, start_month, start_date, tzinfo=timezone.utc)
    end_dt = datetime(end_year, end_month, end_date, tzinfo=timezone.utc)
    in_date_range = (timestamps >= start_dt) & (timestamps < end_dt)
    
    # --- Combined Entry Conditions ---
    long_cond = qqe_long_cond & htf_bull & in_date_range
    short_cond = qqe_short_cond & htf_bear & in_date_range
    
    # --- Generate Entries ---
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_cond.iloc[i] and not pd.isna(close.iloc[i]):
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
        elif short_cond.iloc[i] and not pd.isna(close.iloc[i]):
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
    
    return entries