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
    trade_num = 1

    # Calculate EMAs (current chart)
    ema5 = df['close'].ewm(span=5, adjust=False).mean()
    ema9 = df['close'].ewm(span=9, adjust=False).mean()
    ema13 = df['close'].ewm(span=13, adjust=False).mean()
    ema15 = df['close'].ewm(span=15, adjust=False).mean()

    # EMA fan/stack check
    bullStack = (ema5 > ema9) & (ema9 > ema13) & (ema13 > ema15)
    bearStack = (ema5 < ema9) & (ema9 < ema13) & (ema13 < ema15)

    # Higher timeframe bias (using daily as proxy)
    htfEMA200 = df['close'].ewm(span=200, adjust=False).mean()
    htfBB_basis = df['close'].rolling(20).mean()

    htfBull = (df['close'] > htfEMA200) & (df['close'] > htfBB_basis)
    htfBear = (df['close'] < htfEMA200) & (df['close'] < htfBB_basis)

    allBull = bullStack & htfBull
    allBear = bearStack & htfBear

    # Time filter: 03:00-06:30 America/New_York
    dt_utc = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('America/New_York')
    nyHour = dt_utc.dt.hour
    nyMinute = dt_utc.dt.minute
    nyTotalMins = nyHour * 60 + nyMinute
    inNYSession = (nyTotalMins >= 180) & (nyTotalMins <= 390)

    # FVG detection (3-candle ICT-style)
    bullFVG = df['low'] > df['high'].shift(2)
    bearFVG = df['high'] < df['low'].shift(2)

    # Precompute shifted values
    low_prev1 = df['low'].shift(1)
    high_prev1 = df['high'].shift(1)
    low_prev2 = df['low'].shift(2)
    high_prev2 = df['high'].shift(2)

    # State variables
    state = {
        'bulltap': 0,
        'beartap': 0,
        'bullfvghigh': np.nan,
        'bearfvglow': np.nan
    }

    for i in range(2, len(df)):
        if not inNYSession.iloc[i]:
            continue

        if np.isnan(allBull.iloc[i]) or np.isnan(allBear.iloc[i]):
            continue

        # Bullish FVG detection
        if inNYSession.iloc[i] and allBull.iloc[i] and bullFVG.iloc[i]:
            state['bullfvghigh'] = (df['low'].iloc[i] + high_prev2.iloc[i]) / 2.0
            state['bulltap'] = 0

        # Bearish FVG detection
        if inNYSession.iloc[i] and allBear.iloc[i] and bearFVG.iloc[i]:
            state['bearfvglow'] = (df['high'].iloc[i] + low_prev2.iloc[i]) / 2.0
            state['beartap'] = 0

        # Bull cross detection
        if not np.isnan(state['bullfvghigh']):
            bull_cross = (df['low'].iloc[i] < state['bullfvghigh']) and (low_prev1.iloc[i] >= state['bullfvghigh'])
            if bull_cross:
                state['bulltap'] += 1
                state['beartap'] = 0

        # Bear cross detection
        if not np.isnan(state['bearfvglow']):
            bear_cross = (df['high'].iloc[i] > state['bearfvglow']) and (high_prev1.iloc[i] <= state['bearfvglow'])
            if bear_cross:
                state['beartap'] += 1
                state['bulltap'] = 0

        # Long entry condition
        long_condition = (inNYSession.iloc[i] and allBull.iloc[i] and
                         state['bulltap'] == 1 and not np.isnan(state['bullfvghigh']) and
                         (df['low'].iloc[i] < state['bullfvghigh']) and (low_prev1.iloc[i] >= state['bullfvghigh']))

        if long_condition:
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            state['bulltap'] = 0

        # Short entry condition
        short_condition = (inNYSession.iloc[i] and allBear.iloc[i] and
                          state['beartap'] == 1 and not np.isnan(state['bearfvglow']) and
                          (df['high'].iloc[i] > state['bearfvglow']) and (high_prev1.iloc[i] <= state['bearfvglow']))

        if short_condition:
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            state['beartap'] = 0

    return entries