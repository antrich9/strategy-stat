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
    results = []
    trade_num = 0

    n = len(df)
    if n < 5:
        return results

    # Extract series
    times = df['time'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    # Convert timestamps to datetime for day/week calculations
    dts = [datetime.fromtimestamp(t, tz=timezone.utc) for t in times]

    # Initialize arrays for PDH, PDL (Previous Day High/Low)
    pdh = np.full(n, np.nan)
    pdl = np.full(n, np.nan)
    equilibrium = np.full(n, np.nan)

    # Weekly high/low
    weekly_high = np.full(n, np.nan)
    weekly_low = np.full(n, np.nan)

    # State variables - track per-bar values
    pdh_crossed = np.zeros(n, dtype=bool)
    pdl_crossed = np.zeros(n, dtype=bool)
    bias = [''] * n
    confirmed_bias = [''] * n

    # iFVG state
    ig_active = np.zeros(n, dtype=bool)
    ig_c1_high = np.full(n, np.nan)
    ig_c1_low = np.full(n, np.nan)
    ig_direction = np.zeros(n, dtype=int)
    ig_validation_end = np.full(n, np.nan)

    # Daily swing tracking
    daily_swing_high = np.full(n, np.nan)
    daily_swing_low = np.full(n, np.nan)
    swing_start_bar = np.full(n, np.nan)

    # Process bars
    for i in range(1, n):
        dt = dts[i]
        dt_prev = dts[i-1]

        # Check for new day
        is_new_day = dt.day != dt_prev.day

        # PDH/PDL logic - these come from previous day
        # Since we're iterating chronologically, we need to look ahead or compute differently
        # For simplicity, we'll compute at start of each day using the previous day's H/L
        if is_new_day:
            # Find the previous day's high and low
            day_start = i
            # Look back to find the start of previous day
            prev_day_end = i - 1
            prev_day_start = i - 1
            while prev_day_start > 0 and dts[prev_day_start - 1].day == dts[prev_day_start].day:
                prev_day_start -= 1

            if prev_day_start < prev_day_end:
                pdh[i] = np.max(highs[prev_day_start:prev_day_end + 1])
                pdl[i] = np.min(lows[prev_day_start:prev_day_end + 1])
            else:
                pdh[i] = pdh[i-1] if not np.isnan(pdh[i-1]) else highs[i-1]
                pdl[i] = pdl[i-1] if not np.isnan(pdl[i-1]) else lows[i-1]
        else:
            pdh[i] = pdh[i-1]
            pdl[i] = pdl[i-1]

        # Equilibrium
        if not np.isnan(pdh[i]) and not np.isnan(pdl[i]):
            equilibrium[i] = (pdh[i] + pdl[i]) / 2

        # Reset crossed flags at new day
        if is_new_day:
            pdh_crossed[i] = False
            pdl_crossed[i] = False
            bias[i] = ''
            confirmed_bias[i] = ''
            # Reset daily swing
            daily_swing_high[i] = highs[i]
            daily_swing_low[i] = lows[i]
            swing_start_bar[i] = i
        else:
            pdh_crossed[i] = pdh_crossed[i-1]
            pdl_crossed[i] = pdl_crossed[i-1]
            bias[i] = bias[i-1]
            confirmed_bias[i] = confirmed_bias[i-1]
            daily_swing_high[i] = daily_swing_high[i-1]
            daily_swing_low[i] = daily_swing_low[i-1]
            swing_start_bar[i] = swing_start_bar[i-1]

        # Update daily swing high/low
        if not is_new_day:
            daily_swing_high[i] = max(daily_swing_high[i], highs[i])
            daily_swing_low[i] = min(daily_swing_low[i], lows[i])

        # Check for PDH/PDL crosses
        if not pdh_crossed[i] and closes[i] > pdh[i] and not np.isnan(pdh[i]):
            pdh_crossed[i] = True
            bias[i] = 'buy'

        if not pdl_crossed[i] and closes[i] < pdl[i] and not np.isnan(pdl[i]):
            pdl_crossed[i] = True
            bias[i] = 'sell'

        # Set confirmed bias if noon or later
        is_noon_or_later = dt.hour >= 12
        if not pdh_crossed[i] and not pdl_crossed[i]:
            if is_noon_or_later and confirmed_bias[i] == '':
                if closes[i] > equilibrium[i] and not np.isnan(equilibrium[i]):
                    confirmed_bias[i] = 'BIAS UNCONFIRMED: MODERATELY BULLISH'
                else:
                    confirmed_bias[i] = 'BIAS UNCONFIRMED: MODERATELY BEARISH'

        # iFVG Detection (needs 2 bars back)
        if i >= 2:
            # Bullish FVG: high[2] < low[1] (bar 2 ago's high < bar 1 ago's low)
            bullishFVG1 = highs[i-2] < lows[i-1]
            # Bearish FVG: low[2] > high[1]
            bearishFVG1 = lows[i-2] > highs[i-1]

            if bullishFVG1 and not ig_active[i-1]:
                ig_active[i] = True
                ig_direction[i] = 1
                ig_c1_high[i] = highs[i-2]
                ig_c1_low[i] = lows[i-2]
                ig_validation_end[i] = i + 4
            elif bearishFVG1 and not ig_active[i-1]:
                ig_active[i] = True
                ig_direction[i] = -1
                ig_c1_high[i] = highs[i-2]
                ig_c1_low[i] = lows[i-2]
                ig_validation_end[i] = i + 4
            else:
                ig_active[i] = ig_active[i-1]
                ig_direction[i] = ig_direction[i-1]
                ig_c1_high[i] = ig_c1_high[i-1]
                ig_c1_low[i] = ig_c1_low[i-1]
                ig_validation_end[i] = ig_validation_end[i-1]
        else:
            ig_active[i] = ig_active[i-1] if i > 0 else False
            ig_direction[i] = ig_direction[i-1] if i > 0 else 0
            ig_c1_high[i] = ig_c1_high[i-1] if i > 0 else np.nan
            ig_c1_low[i] = ig_c1_low[i-1] if i > 0 else np.nan
            ig_validation_end[i] = ig_validation_end[i-1] if i > 0 else np.nan

        # Validation check
        validated = False
        if ig_active[i] and i <= ig_validation_end[i]:
            if ig_direction[i] == 1 and closes[i] < ig_c1_high[i]:
                validated = True
            if ig_direction[i] == -1 and closes[i] > ig_c1_low[i]:
                validated = True

        # Entry conditions
        system_entry_long = validated and ig_direction[i] == -1
        system_entry_short = validated and ig_direction[i] == 1

        # Execute entries based on bias
        if system_entry_long and bias[i] == 'sell':
            trade_num += 1
            entry_ts = int(times[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(closes[i])

            results.append({
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

        if system_entry_short and bias[i] == 'buy':
            trade_num += 1
            entry_ts = int(times[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(closes[i])

            results.append({
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

        # Reset ig_active if validated
        if validated:
            ig_active[i] = False

    return results