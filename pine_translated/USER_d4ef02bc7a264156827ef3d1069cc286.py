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
    
    # Strategy parameters (hardcoded from Pine Script inputs)
    atrLength = 14
    emaLength = 200
    lengthK = 8
    smoothK = 5
    lengthD = 3
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    tradeDirection = 'Both'
    input_breakout = True
    input_retest = True
    input_repType = 'On'
    
    # Helper: Wilder RSI
    def wilder_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    # Helper: Wilder ATR
    def wilder_atr(high, low, close, period):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        return atr
    
    # Calculate indicators
    close = df['close']
    high = df['high']
    low = df['low']
    
    ema200 = close.ewm(span=emaLength, adjust=False).mean()
    
    # DMI (ADX)
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr = wilder_atr(high, low, close, 14)
    atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    
    # DiNapoli Stochastic
    lowestLow = low.rolling(lengthK).min()
    highestHigh = high.rolling(lengthK).max()
    stochK = ((close - lowestLow) / (highestHigh - lowestLow) * 100).ewm(span=smoothK, adjust=False).mean()
    stochD = stochK.ewm(span=lengthD, adjust=False).mean()
    
    # Get previous day high/low (using daily lookback)
    prevDayHigh = high.rolling(window=len(df)).apply(lambda x: x.iloc[-1] if len(x) >= 1 else np.nan, raw=False)
    prevDayLow = low.rolling(window=len(df)).apply(lambda x: x.iloc[-1] if len(x) >= 1 else np.nan, raw=False)
    
    # For a proper implementation, we need actual daily data
    # Since we don't have daily data, we'll use a proxy: use rolling max/min of past N bars
    # This is an approximation since request.security doesn't work in this context
    pdh_proxy = high.rolling(288).max().shift(1)  # Approximate daily high
    pdl_proxy = low.rolling(288).min().shift(1)    # Approximate daily low
    
    # Detect breakouts
    breakoutLong = (high > pdh_proxy) & (close > pdh_proxy)
    breakoutShort = (low < pdl_proxy) & (close < pdl_proxy)
    
    # Detect sweeps
    pdhSwept = high > pdh_proxy
    pdlSwept = low < pdl_proxy
    
    # Trend direction
    bullishTrend = close > ema200
    bearishTrend = close < ema200
    
    # Pivot points for support/resistance boxes
    bb = input_lookback
    
    # Calculate pivot low and pivot high
    pl = low.rolling(window=bb+1, min_periods=bb+1).apply(lambda x: x.iloc[bb] if len(x) > bb else np.nan, raw=False)
    ph = high.rolling(window=bb+1, min_periods=bb+1).apply(lambda x: x.iloc[bb] if len(x) > bb else np.nan, raw=False)
    
    # Fill NaN with previous values (fixnan behavior)
    pl = pl.ffill()
    ph = ph.ffill()
    
    # Box top/bottom values
    s_yLoc = np.where(low.shift(bb + 1) > low.shift(bb - 1), low.shift(bb - 1), low.shift(bb + 1))
    r_yLoc = np.where(high.shift(bb + 1) > high.shift(bb - 1), high.shift(bb + 1), high.shift(bb - 1))
    
    sTop = pl
    sBot = pd.Series(s_yLoc, index=low.index)
    rTop = pd.Series(r_yLoc, index=high.index)
    rBot = ph
    
    # Repainting logic
    rTon = input_repType == 'On'
    
    # Breakout detection using crossover/crossunder
    cu_series = close < sBot  # crossunder close below sBot (support broken)
    co_series = close > rTop  # crossover close above rTop (resistance broken)
    
    # Track breakout state
    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)
    
    # Retest conditions (simplified for entry detection)
    sRetValid = pd.Series(False, index=df.index)
    rRetValid = pd.Series(False, index=df.index)
    
    trade_num = 1
    entries = []
    
    # Process bars
    for i in range(bb + 2, len(df)):
        # Update breakout flags
        if i > 0:
            if cu_series.iloc[i] and not cu_series.iloc[i-1]:
                sBreak.iloc[i] = True
            if co_series.iloc[i] and not co_series.iloc[i-1]:
                rBreak.iloc[i] = True
        
        # Long entry logic: breakout above resistance followed by retest to support
        # Entry on retest validation after breakout
        if sBreak.iloc[i] and (i > 0 and not sBreak.iloc[i-1]):
            # Check for retest conditions
            s1 = high.iloc[i] >= sTop.iloc[i] and close.iloc[i] <= sBot.iloc[i]
            s2 = high.iloc[i] >= sTop.iloc[i] and close.iloc[i] >= sBot.iloc[i] and close.iloc[i] <= sTop.iloc[i]
            s3 = high.iloc[i] >= sBot.iloc[i] and high.iloc[i] <= sTop.iloc[i]
            s4 = high.iloc[i] >= sBot.iloc[i] and high.iloc[i] <= sTop.iloc[i] and close.iloc[i] < sBot.iloc[i]
            
            retActive_long = s1 or s2 or s3 or s4
            bars_since_breakout = 0
            for j in range(1, input_retSince + 1):
                if i - j >= 0 and sBreak.iloc[i - j]:
                    break
                bars_since_breakout = j
            
            if retActive_long and bars_since_breakout >= input_retSince:
                if (tradeDirection in ['Long', 'Both']) and input_retest:
                    entry_price = close.iloc[i]
                    entry_ts = df['time'].iloc[i]
                    entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                    
                    entry = {
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
                    }
                    entries.append(entry)
                    trade_num += 1
        
        # Short entry logic: breakdown below support followed by retest to resistance
        if rBreak.iloc[i] and (i > 0 and not rBreak.iloc[i-1]):
            r1 = low.iloc[i] <= rBot.iloc[i] and close.iloc[i] >= rTop.iloc[i]
            r2 = low.iloc[i] <= rBot.iloc[i] and close.iloc[i] <= rTop.iloc[i] and close.iloc[i] >= rBot.iloc[i]
            r3 = low.iloc[i] <= rTop.iloc[i] and low.iloc[i] >= rBot.iloc[i]
            r4 = low.iloc[i] <= rTop.iloc[i] and low.iloc[i] >= rBot.iloc[i] and close.iloc[i] > rTop.iloc[i]
            
            retActive_short = r1 or r2 or r3 or r4
            bars_since_breakout = 0
            for j in range(1, input_retSince + 1):
                if i - j >= 0 and rBreak.iloc[i - j]:
                    break
                bars_since_breakout = j
            
            if retActive_short and bars_since_breakout >= input_retSince:
                if (tradeDirection in ['Short', 'Both']) and input_retest:
                    entry_price = close.iloc[i]
                    entry_ts = df['time'].iloc[i]
                    entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                    
                    entry = {
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
                    }
                    entries.append(entry)
                    trade_num += 1
    
    return entries