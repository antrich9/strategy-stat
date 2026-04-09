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
    df = df.copy().reset_index(drop=True)
    
    # Parameters
    lookback_bars = 12
    threshold = 0.0
    
    # Detect 4H candles from 15min data
    df['ts_dt'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df['4h_period'] = df['time'] // (4 * 60 * 60 * 1000)
    
    # Aggregate 4H OHLCV
    agg_4h = df.groupby('4h_period').agg({
        'time': 'last',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index(drop=True)
    
    high_4h1 = pd.Series(agg_4h['high'].values, index=df.index)
    low_4h1 = pd.Series(agg_4h['low'].values, index=df.index)
    close_4h1 = pd.Series(agg_4h['close'].values, index=df.index)
    volume_4h1 = pd.Series(agg_4h['volume'].values, index=df.index)
    
    # London timezone
    df['london_dt'] = df['ts_dt'].dt.tz_convert('Europe/London')
    df['hour'] = df['london_dt'].dt.hour
    df['minute'] = df['london_dt'].dt.minute
    in_london_window = (
        ((df['hour'] == 7) & (df['minute'] >= 45)) |
        ((df['hour'] >= 8) & (df['hour'] < 11)) |
        ((df['hour'] == 11) & (df['minute'] < 45)) |
        ((df['hour'] == 14) & (df['minute'] >= 0)) |
        ((df['hour'] == 14) & (df['minute'] < 45))
    )
    
    # New 4H candle detection
    df['prev_4h'] = df['4h_period'].shift(1)
    is_new_4h1 = (df['4h_period'] != df['prev_4h']) & df['prev_4h'].notna()
    
    # Wilder RSI implementation
    def wilder_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Wilder ATR implementation
    def wilder_atr(high, low, close, period):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/period, adjust=False).mean()
        return atr
    
    # Volume filter (4H)
    vol_sma_4h = volume_4h1.rolling(9).mean()
    volfilt1 = volume_4h1.shift(1) > vol_sma_4h.shift(1) * 1.5
    
    # ATR filter (4H)
    atr_4h = wilder_atr(high_4h1, low_4h1, close_4h1, 20)
    atrfilt1 = (low_4h1 - high_4h1.shift(2) > atr_4h / 1.5) | (low_4h1.shift(2) - high_4h1 > atr_4h / 1.5)
    
    # Trend filter (4H)
    loc1 = close_4h1.rolling(54).mean()
    locfiltb1 = loc1 > loc1.shift(1)
    locfilts1 = loc1 <= loc1.shift(1)
    
    # FVG conditions (4H)
    bfvg1 = (low_4h1 > high_4h1.shift(2)) & volfilt1 & atrfilt1 & locfiltb1
    sfvg1 = (high_4h1 < low_4h1.shift(2)) & volfilt1 & atrfilt1 & locfilts1
    
    # Track sharp turns
    lastFVG = 0
    entries = []
    trade_num = 1
    
    # For BPR (chart timeframe)
    bull_fvg1 = (low > high.shift(2)) & (close.shift(1) < low.shift(2))
    bear_fvg1 = (high < low.shift(2)) & (close.shift(1) > high.shift(2))
    
    # Wilder's min/max for BPR
    def wilder_min(series, period):
        result = series.rolling(period).min()
        alpha = 1.0 / period
        for i in range(period, len(series)):
            if pd.notna(result.iloc[i - 1]):
                result.iloc[i] = min(series.iloc[i], result.iloc[i - 1]) if pd.notna(series.iloc[i]) else result.iloc[i - 1]
        return result
    
    def wilder_max(series, period):
        result = series.rolling(period).max()
        alpha = 1.0 / period
        for i in range(period, len(series)):
            if pd.notna(result.iloc[i - 1]):
                result.iloc[i] = max(series.iloc[i], result.iloc[i - 1]) if pd.notna(series.iloc[i]) else result.iloc[i - 1]
        return result
    
    warmup = 54
    bull_result = pd.Series(False, index=df.index)
    bear_result = pd.Series(False, index=df.index)
    
    for i in range(3, len(df)):
        if i < warmup:
            continue
        bull_since = 0
        for j in range(i - 1, max(0, i - lookback_bars - 1), -1):
            if bull_fvg1.iloc[j]:
                bull_since = i - j
                break
        if bull_since > lookback_bars:
            bull_since = lookback_bars + 1
        
        bull_cond_1 = bull_fvg1.iloc[i] and bull_since <= lookback_bars
        if bull_cond_1:
            combined_low = max(high.iloc[i - bull_since], high.iloc[i - 2]) if pd.notna(high.iloc[i - bull_since]) and pd.notna(high.iloc[i - 2]) else np.nan
            combined_high = min(low.iloc[i - bull_since + 2], low.iloc[i]) if pd.notna(low.iloc[i - bull_since + 2]) and pd.notna(low.iloc[i]) else np.nan
            if pd.notna(combined_high) and pd.notna(combined_low):
                bull_result.iloc[i] = (combined_high - combined_low) >= threshold
        
        bear_since = 0
        for j in range(i - 1, max(0, i - lookback_bars - 1), -1):
            if bear_fvg1.iloc[j]:
                bear_since = i - j
                break
        if bear_since > lookback_bars:
            bear_since = lookback_bars + 1
        
        bear_cond_1 = bear_fvg1.iloc[i] and bear_since <= lookback_bars