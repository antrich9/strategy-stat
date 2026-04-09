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
    entries = []
    trade_num = 0
    
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    time = df['time']
    
    # ZigZag calculation parameters
    lookback = 3
    pivot_type = "Close"
    
    def get_pivot_levels(src_series, lb):
        pivots_high = src_series.rolling(window=2*lb+1, center=True).max().shift(-lb)
        pivots_low = src_series.rolling(window=2*lb+1, center=True).min().shift(-lb)
        
        is_pivot_high = (src_series == pivots_high) & (src_series == src_series.rolling(lb+1).max().shift(-lb))
        is_pivot_low = (src_series == pivots_low) & (src_series == src_series.rolling(lb+1).min().shift(-lb))
        
        return pivots_high, pivots_low, is_pivot_high, is_pivot_low
    
    # Calculate pivots based on pivotType
    if pivot_type == "Close":
        pivots_high_c, pivots_low_c, is_ph_c, is_pl_c = get_pivot_levels(close, lookback)
    else:
        pivots_high_c, pivots_low_c, is_ph_c, is_pl_c = get_pivot_levels(high, lookback)
    
    # Wilder RSI implementation
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # ATR calculation (Wilder)
    def wilder_atr(high_series, low_series, close_series, length):
        tr1 = high_series - low_series
        tr2 = abs(high_series - close_series.shift(1))
        tr3 = abs(low_series - close_series.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr
    
    # Calculate RSI for different lengths
    rsi_14 = wilder_rsi(close, 14)
    rsi_7 = wilder_rsi(close, 7)
    
    # Calculate ATR
    atr_14 = wilder_atr(high, low, close, 14)
    
    # Determine timeframe multiplier (assumes lower timeframe is the base)
    # HTF inputs: higher_timeframe_1 = 240, higher_timeframe_2 = 60, higher_timeframe_3 = 15
    # For simulation, we'll create synthetic HTF data by resampling
    # This is a simplification since we don't have actual HTF data
    # We'll use EMA smoothing to approximate HTF trends
    
    # Create approximations for HTF trends using different periods
    # 240 min -> ~4h, 60 min -> 1h, 15 min -> 15min
    # Use different EMA spans to approximate
    
    # For simulation: use ema smoothing to create HTF-like trend indicators
    htf_trend_1 = close.ewm(span=50, adjust=False).mean()  # Approx Daily (240)
    htf_trend_2 = close.ewm(span=20, adjust=False).mean()  # Approx 4H (60)
    htf_trend_3 = close.ewm(span=8, adjust=False).mean()   # Approx 1H (15)
    
    # Detect trend direction for each "HTF"
    htf_close_1_above = htf_trend_1 > htf_trend_1.shift(1)  # Daily bullish
    htf_close_2_above = htf_trend_2 > htf_trend_2.shift(1)  # 4H bullish
    htf_close_3_above = htf_trend_3 > htf_trend_3.shift(1)  # 1H bullish
    
    # Previous trend state
    prev_trend_1 = htf_close_1_above.shift(1).fillna(False)
    prev_trend_2 = htf_close_2_above.shift(1).fillna(False)
    prev_trend_3 = htf_close_3_above.shift(1).fillna(False)
    
    # HTF close > open (bullish candle) for each timeframe
    htf_bull_1 = (close > open_).rolling(50).sum() > 25  # Approximate
    htf_bear_1 = (close < open_).rolling(50).sum() > 25
    
    # Determine trend direction
    trend_1_bullish = htf_close_1_above
    trend_1_bearish = ~htf_close_1_above
    
    trend_2_bullish = htf_close_2_above
    trend_2_bearish = ~htf_close_2_above
    
    trend_3_bullish = htf_close_3_above
    trend_3_bearish = ~htf_close_3_above
    
    # State tracking for color change (wasRed, wasGreen, Green, Red)
    was_red = False
    was_green = False
    green_state = False
    red_state = False
    
    # Pivot levels from higher timeframes (using current timeframe pivots as proxy)
    # f_getHTF() returns: ph, pl, highLevel, lowLevel, barsSinceHigh, barsSinceLow, timeSinceHigh, timeSinceLow
    
    # Use current pivots as HTF levels
    ph_levels = pd.Series(np.nan, index=df.index)
    pl_levels = pd.Series(np.nan, index=df.index)
    
    for i in range(lookback * 2 + 1, len(df)):
        window_high = high.iloc[i-lookback:i+lookback+1].max()
        window_low = low.iloc[i-lookback:i+lookback+1].min()
        if pivot_type == "Close":
            if close.iloc[i] == window_high and close.iloc[i] == high.iloc[i-lookback:i+1].max():
                ph_levels.iloc[i] = close.iloc[i]
            if close.iloc[i] == window_low and close.iloc[i] == low.iloc[i-lookback:i+1].min():
                pl_levels.iloc[i] = close.iloc[i]
        else:
            if high.iloc[i] >= window_high and high.iloc[i] >= high.iloc[i-lookback:i+1].max():
                ph_levels.iloc[i] = high.iloc[i]
            if low.iloc[i] <= window_low and low.iloc[i] <= low.iloc[i-lookback:i+1].min():
                pl_levels.iloc[i] = low.iloc[i]
    
    # Forward fill pivot levels
    ph_levels = ph_levels.ffill()
    pl_levels = pl_levels.ffill()
    
    # Calculate ZigZag swing highs/lows for structure detection
    swing_high = pd.Series(np.nan, index=df.index)
    swing_low = pd.Series(np.nan, index=df.index)
    
    for i in range(lookback, len(df) - lookback):
        left_high = high.iloc[i-lookback:i].max()
        right_high = high.iloc[i+1:i+lookback+1].max()
        left_low = low.iloc[i-lookback:i].min()
        right_low = low.iloc[i+1:i+lookback+1].min()
        
        if high.iloc[i] > left_high and high.iloc[i] >= right_high:
            swing_high.iloc[i] = high.iloc[i]
        if low.iloc[i] < left_low and low.iloc[i] <= right_low:
            swing_low.iloc[i] = low.iloc[i]
    
    # Calculate market structure (MSS - Market Structure Shift)
    # Higher high, higher low = bullish structure
    # Lower high, lower low = bearish structure
    
    # Detect BOS (Break of Structure) and CHoCH (Change of Character)
    
    # Entry conditions based on the strategy name and logic
    # "HTF no engulf fib 0.5 s cont Market Structures + ZigZag"
    # Key components: HTF trend alignment, market structure break, 0.5 fib retracement
    
    # Simplify: Entry when structure breaks in direction of HTF trend
    
    # Track structure states
    structure_bull = False
    structure_bear = False
    
    # Pivot high/low detection for structure
    ph_detected = pd.Series(False, index=df.index)
    pl_detected = pd.Series(False, index=df.index)
    
    # Detect pivots
    for i in range(lookback * 2, len(df) - lookback):
        is_high = True
        is_low = True
        for j in range(1, lookback + 1):
            if high.iloc[i] <= high.iloc[i-j] or high.iloc[i] <= high.iloc[i+j]:
                is_high = False
            if low.iloc[i] >= low.iloc[j] or low.iloc[i] >= low.iloc[i+j]:
                is_low = False
        if is_high:
            ph_detected.iloc[i] = True
        if is_low:
            pl_detected.iloc[i] = True
    
    # Calculate structure breaks
    prev_ph = np.nan
    prev_pl = np.nan
    
    structure_break_bull = pd.Series(False, index=df.index)
    structure_break_bear = pd.Series(False, index=df.index)
    
    for i in range(lookback * 3, len(df)):
        if ph_detected.iloc[i]:
            if not np.isnan(prev_ph) and high.iloc[i] > prev_ph:
                structure_break_bull.iloc[i] = True
            prev_ph = high.iloc[i]
        if pl_detected.iloc[i]:
            if not np.isnan(prev_pl) and low.iloc[i] < prev_pl:
                structure_break_bear.iloc[i] = True
            prev_pl = low.iloc[i]
    
    # Entry signals
    
    # Long entry: HTF trend bullish + structure break + RSI filter
    # Short entry: HTF trend bearish + structure break + RSI filter
    
    long_condition = (
        (trend_1_bullish | trend_2_bullish) &  # Higher timeframe bullish
        structure_break_bull &  # Bullish structure break
        (rsi_14 > 40) &  # RSI above oversold
        (rsi_14 < 70)   # Not overbought
    )
    
    short_condition = (
        (trend_1_bearish | trend_2_bearish) &  # Higher timeframe bearish
        structure_break_bear &  # Bearish structure break
        (rsi_14 < 60) &  # RSI below overbought
        (rsi_14 > 30)   # Not oversold
    )
    
    # Generate entries
    for i in range(len(df)):
        if i < lookback * 3:
            continue
        
        # Check for NaN indicators
        if pd.isna(rsi_14.iloc[i]) or pd.isna(atr_14.iloc[i]):
            continue
        
        entry_price = close.iloc[i]
        entry_ts = int(time.iloc[i])
        entry_time_str = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat() if entry_ts > 1e12 else datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        
        # Long entry
        if long_condition.iloc[i]:
            trade_num += 1
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
        
        # Short entry
        if short_condition.iloc[i]:
            trade_num += 1
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
    
    return entries