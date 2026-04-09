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
    n = len(df)
    if n < 20:
        return []

    # ========== HELPER FUNCTIONS ==========
    def wilder_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - pd.Series(close).shift(1))
        tr3 = np.abs(low - pd.Series(close).shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr

    def wilder_rsi(close, period=14):
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def ta_sma(series, length):
        return series.rolling(length).mean()

    def ta_ema(series, length):
        return series.ewm(span=length, adjust=False).mean()

    def ta_atr(high, low, close, length=14):
        return wilder_atr(high, low, close, length)

    # ========== PINE SCRIPT VARIABLES ==========
    prd = 2
    atrMultiplier = 3.0
    riskPerTrade = 1.0
    Rvalue = 1.0
    inp1 = False  # Volume Filter
    inp2 = False  # ATR Filter
    inp3 = False  # Trend Filter

    # ========== ZIGZAG CALCULATION ==========
    # Simplified ZigZag using swing highs/lows
    zigzag_high = np.nan
    zigzag_low = np.nan
    zigzag_dir = np.zeros(n)  # 0 = undecided, 1 = uptrend, -1 = downtrend

    # Use a simpler approach: identify local extrema
    swing_window = prd

    for i in range(swing_window, n - swing_window):
        # Check for swing high
        is_high = True
        for j in range(1, swing_window + 1):
            if df['high'].iloc[i] <= df['high'].iloc[i - j] or df['high'].iloc[i] <= df['high'].iloc[i + j]:
                is_high = False
                break
        # Check for swing low
        is_low = True
        for j in range(1, swing_window + 1):
            if df['low'].iloc[i] >= df['low'].iloc[i - j] or df['low'].iloc[i] >= df['low'].iloc[i + j]:
                is_low = False
                break

        if is_high and not is_low:
            zigzag_high = df['high'].iloc[i]
        elif is_low and not is_high:
            zigzag_low = df['low'].iloc[i]

    # ========== FIBONACCI LEVELS ==========
    # Find last two zigzag points to calculate fib levels
    fib_0 = np.nan
    fib_50 = np.nan
    fib_1 = np.nan

    # Simplified: use recent swing high/low as fib reference
    if n >= 20:
        recent_high = df['high'].rolling(20).max().iloc[-1]
        recent_low = df['low'].rolling(20).min().iloc[-1]
        diff = recent_high - recent_low
        fib_0 = recent_low
        fib_50 = recent_low + diff * 0.5
        fib_1 = recent_high

    # ========== ATR ==========
    atr1 = ta_atr(df['high'], df['low'], df['close'], 14)

    # ========== FILTERS ==========
    volfilt = df['volume'] > ta_sma(df['volume'], 9) * 1.5 if inp1 else True
    atr2 = ta_atr(df['high'], df['low'], df['close'], 20) / 1.5
    atrfilt = ((df['low'] - df['high'].shift(2) > atr2) | (df['low'].shift(2) - df['high'] > atr2)) if inp2 else True
    loc = ta_sma(df['close'], 54)
    loc2 = loc > loc.shift(1)
    locfiltb = loc2 if inp3 else True
    locfilts = ~loc2 if inp3 else True

    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts

    # ========== TRADING WINDOWS ==========
    london_start_window1 = 7 * 60 + 45  # 07:45 in minutes from midnight
    london_end_window1 = 9 * 60 + 45    # 09:45 in minutes from midnight
    london_start_window2 = 14 * 60 + 45  # 14:45 in minutes from midnight
    london_end_window2 = 16 * 60 + 45   # 16:45 in minutes from midnight

    in_trading_window = np.zeros(n, dtype=bool)
    for i in range(n):
        dt = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc)
        minutes = dt.hour * 60 + dt.minute
        in_window1 = london_start_window1 <= minutes < london_end_window1
        in_window2 = london_start_window2 <= minutes < london_end_window2
        in_trading_window[i] = in_window1 or in_window2

    # ========== FVG DETECTION ==========
    # Bullish FVG: current low > high 2 bars ago AND previous bar was bearish
    obUp1 = (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open']) & (df['close'] > df['high'].shift(1))
    obDown1 = (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open']) & (df['close'] < df['low'].shift(1))
    fvgUp1 = df['low'] > df['high'].shift(2)
    fvgDown1 = df['high'] < df['low'].shift(2)

    # Detect FVG patterns
    bullishFVG1 = (obUp1 & fvgUp1 & (df['close'] < fib_50) & (df['close'] > fib_0)) if not np.isnan(fib_50) else False
    bearishFVG1 = (obDown1 & fvgDown1 & (df['close'] > fib_50) & (df['close'] < fib_1)) if not np.isnan(fib_50) else False

    # Create FVG high/low signals
    bullfvghigh1 = np.where(bullishFVG1, df['low'], np.nan)
    bearfvglow1 = np.where(bearishFVG1, df['high'], np.nan)

    # Forward fill to propagate FVG levels
    bullfvghigh1_series = pd.Series(bullfvghigh1).ffill()
    bearfvglow1_series = pd.Series(bearfvglow1).ffill()

    # ========== TAP COUNTING ==========
    bulltap1 = np.zeros(n)
    beartap1 = np.zeros(n)

    bulltap_count = 0
    beartap_count = 0

    for i in range(1, n):
        # Bull tap: price crosses below bullfvghigh1
        if not np.isnan(bullfvghigh1_series.iloc[i-1]):
            if df['low'].iloc[i] < bullfvghigh1_series.iloc[i-1] and df['low'].iloc[i-1] >= bullfvghigh1_series.iloc[i-1]:
                bulltap_count += 1
                beartap_count = 0
        bulltap1[i] = bulltap_count

        # Bear tap: price crosses above bearfvglow1
        if not np.isnan(bearfvglow1_series.iloc[i-1]):
            if df['high'].iloc[i] > bearfvglow1_series.iloc[i-1] and df['high'].iloc[i-1] <= bearfvglow1_series.iloc[i-1]:
                beartap_count += 1
                bulltap_count = 0
        beartap1[i] = beartap_count

    # ========== ENTRY CONDITIONS ==========
    # Long conditions
    longCondition1 = (df['low'] < bullfvghigh1_series) & (df['low'].shift(1) >= bullfvghigh1_series) & ~np.isnan(bullfvghigh1_series)
    longCondition2 = bulltap1 == 1
    longCondition3 = in_trading_window
    longCondition4 = df['close'] > fib_0 if not np.isnan(fib_0) else False
    longCondition5 = ~np.isnan(fib_50) & ~np.isnan(fib_0) & ~np.isnan(fib_1)
    longAllConditions = longCondition1 & longCondition2 & longCondition3 & longCondition4 & longCondition5

    # Short conditions
    shortCondition1 = (df['high'] > bearfvglow1_series) & (df['high'].shift(1) <= bearfvglow1_series) & ~np.isnan(bearfvglow1_series)
    shortCondition2 = beartap1 == 1
    shortCondition3 = in_trading_window
    shortCondition4 = df['close'] < fib_1 if not np.isnan(fib_1) else False
    shortCondition5 = ~np.isnan(fib_50) & ~np.isnan(fib_0) & ~np.isnan(fib_1)
    shortAllConditions = shortCondition1 & shortCondition2 & shortCondition3 & shortCondition4 & shortCondition5

    # ========== GENERATE ENTRIES ==========
    entries = []
    trade_num = 1

    for i in range(n):
        if i < 2:
            continue

        if longAllConditions.iloc[i]:
            entry_price = df['close'].iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()

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

        if shortAllConditions.iloc[i]:
            entry_price = df['close'].iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()

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