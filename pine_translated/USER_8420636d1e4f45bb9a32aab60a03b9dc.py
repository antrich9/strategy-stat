import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Parameters
    PP = 7  # Pivot period
    # We can also compute other parameters like ATR, but not needed for entry

    # Compute pivot highs and lows
    # Use rolling windows to find local peaks
    # We can compute pivot high: high[i] > high[i-1] and high[i] > high[i+1] for a window of PP

    # However, we need to consider the forward window. In Pine Script, pivot high is confirmed after PP bars.
    # In Python, we can compute a pivot high series by checking past and future values.

    # We can compute pivot high as: for each i, if high[i] == max(high[i-PP:i+PP+1]) and high[i] > high[i-PP] and high[i] > high[i+PP]
    # But we need to handle boundaries.

    # Let's compute pivot highs and lows using a simple method:
    # For each i, we look back PP bars and forward PP bars.

    high = df['high']
    low = df['low']
    close = df['close']
    ts = df['time']

    # Initialize pivot high and low series
    pivot_high = pd.Series(False, index=df.index)
    pivot_low = pd.Series(False, index=df.index)

    # Compute pivot high and low for bars where we have enough history
    for i in range(PP, len(df) - PP):
        # Check if high[i] is the highest in the window
        if high.iloc[i] == high.iloc[i-PP:i+PP+1].max() and high.iloc[i] > high.iloc[i-PP] and high.iloc[i] > high.iloc[i+PP]:
            pivot_high.iloc[i] = True
        if low.iloc[i] == low.iloc[i-PP:i+PP+1].min() and low.iloc[i] < low.iloc[i-PP] and low.iloc[i] < low.iloc[i+PP]:
            pivot_low.iloc[i] = True

    # Now we need to find major pivots. We can take the last pivot high and low and track them.
    # We'll track the most recent major pivot high and low.

    # We also need to compute the BoS condition: price breaks above the previous pivot high (for bullish) or below the previous pivot low (for bearish).

    # We'll track the previous pivot high and low values.

    # We'll also need to handle the case where there is a series of pivots.

    # Let's create a series to hold the pivot high value at each pivot, and similarly for pivot low.

    # We'll create a series that holds the pivot high value only at pivot high bars, else NaN.

    pivot_high_val = pd.Series(np.nan, index=df.index)
    pivot_low_val = pd.Series(np.nan, index=df.index)

    pivot_high_val[pivot_high] = high[pivot_high]
    pivot_low_val[pivot_low] = low[pivot_low]

    # Forward fill to get the most recent pivot high/low up to each bar.
    # However, we only want to consider the previous pivot, not the current one.

    # We can shift the series by 1 to get the previous pivot.

    prev_pivot_high = pivot_high_val.shift(1)
    prev_pivot_low = pivot_low_val.shift(1)

    # Now we can compute the break conditions:
    # Bullish BoS: close > prev_pivot_high and prev_pivot_high is not NaN
    # Bearish BoS: close < prev_pivot_low and prev_pivot_low is not NaN

    # However, we need to ensure that the prev_pivot_high is a pivot high, not just any high.

    # Let's compute the break conditions as boolean series.

    bullish_bos = (close > prev_pivot_high) & prev_pivot_high.notna()
    bearish_bos = (close < prev_pivot_low) & prev_pivot_low.notna()

    # Now we can generate entries when these conditions are true.

    # However, we also need to consider the "LockBreak_M" variable. This might be used to avoid false breakouts.
    # We can ignore it for simplicity.

    # Now generate entries.

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if bullish_bos.iloc[i]:
            entry_ts = int(ts.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
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
        elif bearish_bos.iloc[i]:
            entry_ts = int(ts.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
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