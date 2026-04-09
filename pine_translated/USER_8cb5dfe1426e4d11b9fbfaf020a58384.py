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
    import pandas as pd
    import numpy as np
    from datetime import datetime, timezone

    # Ensure df is sorted by time
    df = df.sort_values('time').reset_index(drop=True)

    # Initialize variables
    trade_num = 0
    entries = []

    # Precompute shift values
    low = df['low']
    high = df['high']
    close = df['close']

    # Compute bull_fvg and bear_fvg conditions
    # bull_fvg: low > high[2] and close[1] > high[2]
    # bear_fvg: high < low[2] and close[1] < low[2]
    bull_fvg = (low > high.shift(2)) & (close.shift(1) > high.shift(2))
    bear_fvg = (high < low.shift(2)) & (close.shift(1) < low.shift(2))

    # Initialize consecutive counters and last timestamp
    consecutive_bull = 0
    consecutive_bear = 0
    last_t = None

    # Iterate over rows
    for i in range(len(df)):
        ts = int(df['time'].iloc[i])

        # Check for bull_fvg
        if bull_fvg.iloc[i]:
            # Avoid processing same timestamp twice
            if ts != last_t:
                # Increment consecutive bull count, reset bear
                consecutive_bull += 1
                consecutive_bear = 0
                last_t = ts

                # If we have at least 2 consecutive bull FVGs, generate entry
                if consecutive_bull >= 2:
                    # Midpoint of the second FVG: (low[i] + high[i-2]) / 2
                    midpoint = (low.iloc[i] + high.iloc[i-2]) / 2.0
                    trade_num += 1
                    entry = {
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': ts,
                        'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                        'entry_price_guess': midpoint,
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': midpoint,
                        'raw_price_b': midpoint
                    }
                    entries.append(entry)
            else:
                # Same timestamp as previous FVG, skip (or could reset counters)
                pass

        # Check for bear_fvg
        elif bear_fvg.iloc[i]:
            if ts != last_t:
                consecutive_bear += 1
                consecutive_bull = 0
                last_t = ts

                if consecutive_bear >= 2:
                    # Midpoint of the second bear FVG: (low[i-2] + high[i]) / 2
                    midpoint = (low.iloc[i-2] + high.iloc[i]) / 2.0
                    trade_num += 1
                    entry = {
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': ts,
                        'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                        'entry_price_guess': midpoint,
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': midpoint,
                        'raw_price_b': midpoint
                    }
                    entries.append(entry)
            else:
                pass
        else:
            # No FVG on this bar, reset consecutive counts? Actually script resets only when opposite appears.
            # In script, when bull_fvg occurs, consecutiveBullCount increments and consecutiveBearCount resets.
            # When bear_fvg occurs, vice versa.
            # If no FVG, they don't reset. Actually they only reset when opposite appears.
            # But we can keep counts across bars; they only increment when FVG appears.
            # So we don't reset here.
            pass

    return entries