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
    lookback = 3
    pivotType = 'Close'
    oneSet = 'D'
    twoSet = '240'
    threeSet = '60'

    close = df['close']
    high = df['high']
    low = df['low']

    # Helper function to approximate higher timeframe data
    def get_htf_data(series, tf_str):
        if tf_str == 'D':
            # Daily: approximately 1440 1m bars
            period = 1440
        elif tf_str == '240':
            # 4H: approximately 240 1m bars
            period = 240
        elif tf_str == '60':
            # 1H: approximately 60 1m bars
            period = 60
        else:
            period = 100
        
        # Use rolling to approximate higher timeframe values
        # Take the last value in each window as the "close" for that HTF bar
        rolling_max = series.rolling(period, min_periods=1).max()
        rolling_min = series.rolling(period, min_periods=1).min()
        rolling_close = series.rolling(period, min_periods=1).mean()
        rolling_open = series.rolling(period, min_periods=1).mean()
        
        # Shift to align (last value in window represents completed HTF bar)
        htf_open = rolling_open.shift(period - 1)
        htf_close = rolling_close.shift(period - 1)
        htf_high = rolling_max.shift(period - 1)
        htf_low = rolling_min.shift(period - 1)
        
        return htf_close, htf_open, htf_high, htf_low

    # Get higher timeframe data
    htf_close_1, htf_open_1, htf_high_1, htf_low_1 = get_htf_data(close, oneSet)
    htf_close_2, htf_open_2, htf_high_2, htf_low_2 = get_htf_data(close, twoSet)
    htf_close_3, htf_open_3, htf_high_3, htf_low_3 = get_htf_data(close, threeSet)

    # Determine trend based on close > open for each timeframe
    trend_1 = (htf_close_1 > htf_open_1).astype(float).fillna(0)
    trend_2 = (htf_close_2 > htf_open_2).astype(float).fillna(0)
    trend_3 = (htf_close_3 > htf_open_3).astype(float).fillna(0)

    # Combined trend signal (majority agreement with stronger weight for higher TF)
    # Daily (D) has weight 3, 4H has weight 2, 1H has weight 1
    combined_trend = (trend_1 * 3 + trend_2 * 2 + trend_3 * 1) / 6.0

    # Wilder RSI implementation
    def wilder_rsi(src, length):
        delta = src.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Wilder ATR implementation
    def wilder_atr(high, low, close, length):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr

    # Calculate pivot levels
    def calc_pivots(close, high, low, lookback, pivotType):
        if pivotType == 'Close':
            ph = close.rolling(2*lookback+1).apply(lambda x: x.max() if x.name == x.idxmax() else np.nan, raw=False)
            pl = close.rolling(2*lookback+1).apply(lambda x: x.min() if x.name == x.idxmin() else np.nan, raw=False)
        else:
            ph = high.rolling(2*lookback+1).apply(lambda x: x.max() if x.name == x.idxmax() else np.nan, raw=False)
            pl = low.rolling(2*lookback+1).apply(lambda x: x.min() if x.name == x.idxmin() else np.nan, raw=False)
        
        ph = ph.replace({np.nan: np.nan})
        pl = pl.replace({np.nan: np.nan})
        
        # High/low levels using rolling max/min centered on pivot
        if pivotType == 'Close':
            highLevel = close.rolling(2*lookback+1).max().shift(lookback)
            lowLevel = close.rolling(2*lookback+1).min().shift(lookback)
        else:
            highLevel = high.rolling(2*lookback+1).max().shift(lookback)
            lowLevel = low.rolling(2*lookback+1).min().shift(lookback)
        
        return ph, pl, highLevel, lowLevel

    ph, pl, highLevel, lowLevel = calc_pivots(close, high, low, lookback, pivotType)

    # Calculate RSI and ATR
    rsi = wilder_rsi(close, 14)
    atr = wilder_atr(high, low, close, 14)

    # Detect FVG (Fair Value Gap) - candle with large body relative to wicks
    # Bullish FVG: current candle body > 1.5x ATR and close > open, prev candle had opposite
    fvg_bullish = ((close - open) > 1.5 * atr) & (close > open) & (close.shift(1) < open.shift(1))
    # Bearish FVG: current candle body > 1.5x ATR and close < open, prev candle had opposite
    fvg_bearish = ((open - close) > 1.5 * atr) & (close < open) & (close.shift(1) > open.shift(1))

    # Uptrend/Downtrend signal based on price action vs pivot levels
    uptrendSignal = (high > highLevel).astype(float)
    downtrendSignal = (low < lowLevel).astype(float)

    # Track trend state with crossover detection
    # Long: uptrendSignal crosses above 0 (entering uptrend)
    # Short: downtrendSignal crosses above 0 (entering downtrend) or uptrendSignal drops to 0

    # Use combined trend to determine direction
    # Combined trend > 0.5 means bullish bias
    trend_bullish = combined_trend > 0.5

    # Entry conditions
    # Long entry: trend changes to bullish AND price above recent low (bullish confirmation)
    long_condition = (trend_bullish) & (trend_bullish.shift(1) == False)
    
    # Short entry: trend changes to bearish AND price below recent high (bearish confirmation)
    short_condition = (~trend_bullish) & (trend_bullish.shift(1) == True)

    # Also check FVG for entry confirmation
    long_condition = long_condition & (fvg_bullish | (close > close.shift(lookback)))
    short_condition = short_condition & (fvg_bearish | (close < close.shift(lookback)))

    # Additional: Check RSI for overbought/oversold confirmation
    long_condition = long_condition & (rsi < 70)  # Not overbought for long
    short_condition = short_condition & (rsi > 30)  # Not oversold for short

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(1, len(df)):
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = df['close'].iloc[i]

        # Check for NaN in required indicators
        if pd.isna(highLevel.iloc[i]) or pd.isna(lowLevel.iloc[i]):
            continue
        if pd.isna(combined_trend.iloc[i]) or pd.isna(rsi.iloc[i]):
            continue

        # Check long condition with crossover logic
        # Crossover: trend just turned bullish
        if long_condition.iloc[i] and not long_condition.iloc[i-1]:
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

        # Check short condition with crossover logic
        # Crossunder: trend just turned bearish
        if short_condition.iloc[i] and not short_condition.iloc[i-1]:
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