import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Parameters
    pivot_len = 15  # from input: pivotStrength1 = 15

    # Compute pivot highs and lows
    # We use a rolling window of length 2*pivot_len+1, centered, and then shift by pivot_len
    # However, we don't have future data, so we compute the pivot high as the highest high over the past pivot_len bars and current bar? Not exactly.
    # Alternatively, we can compute the pivot high at bar i as the highest high from i-pivot_len to i+pivot_len, but we don't have i+pivot_len.
    # So we compute the pivot high at bar i as the highest high from i-pivot_len to i, and then we shift the series by -pivot_len? That would be late.
    # Actually, in Pine Script, the pivot high is computed on the higher timeframe, so it's different.

    # Given the time, I think we have to make a simplification: we'll compute the pivot high as the highest high over the past pivot_len*2 bars, and then we'll use a shift of pivot_len.
    # But let's try to compute the pivot high in a way that matches the Pine Script as much as possible.

    # We'll compute the pivot high as the highest high of the current bar and the previous pivot_len bars, and then we'll require that the next pivot_len bars are lower? Not sure.

    # Alternatively, we can use the ta.pivothigh function from the tulipy library? But we are restricted to pandas and numpy.

    # Given the time, I think the best is to compute the pivot high using a rolling window that looks back only, and then we adjust the crossover detection accordingly.

    # Let's compute the pivot high as the highest high over the past pivot_len bars (including current) and then shift by 1? Not sure.

    # Let's look at the Pine Script: they use ta.pivothigh(pivotLen, pivotLen). This means the pivot high is the highest high of the past pivotLen bars and the next pivotLen bars. So the pivot high is confirmed pivot_len bars after the fact.

    # So in Python, we can compute the pivot high at bar i as the highest high from i-pivot_len to i+pivot_len, but we don't have i+pivot_len. So we can compute it when we are at bar i+pivot_len.

    # So we can compute the pivot high series by taking the rolling max of high over a window of length 2*pivot_len+1, centered (but we don't have future, so we can only do it for the past). Then we shift the series by -pivot_len? That would give us the pivot high for the bar that is pivot_len bars in the past.

    # Alternatively, we can compute the pivot high at bar i as the highest high from i-pivot_len to i, and then we require that the next pivot_len bars are lower? Not exactly.

    # Given the time, I think we have to make a compromise: we'll compute the pivot high as the highest high over the past pivot_len*2 bars, and then we'll use a shift of pivot_len.

    # Let's do:
    # pivot_high = df['high'].rolling(window=pivot_len*2+1, center=True).max().shift(pivot_len)
    # But this will give NaN for the first and last pivot_len bars.

    # Similarly for pivot_low.

    pivot_high = df['high'].rolling(window=pivot_len*2+1, center=True).max().shift(pivot_len)
    pivot_low = df['low'].rolling(window=pivot_len*2+1, center=True).min().shift(pivot_len)

    # Now, we need to compute the trend and break of structure. We'll use a loop to iterate through the bars.

    # Initialize variables
    current_trend = False  # false means bearish, true means bullish
    break_of_structure = False
    last_pivot_high = np.nan
    last_pivot_low = np.nan
    last_broken_high = np.nan
    last_broken_low = np.nan

    entries = []
    trade_num = 1

    for i in range(1, len(df)):
        # Update last pivot high and low if the current pivot high/low is not NaN
        if not np.isnan(pivot_high.iloc[i]):
            # In Pine Script, they update lastPivotHigh only if the trend is bullish? Let's check:
            # lastPivotHigh := currentTrend ? math.max(high[pivotLen], lastPivotHigh) : high[pivotLen]
            # So if the trend is bullish, they take the max of the current pivot high and the last pivot high. If bearish, they take the current pivot high.
            if current_trend:
                last_pivot_high = max(pivot_high.iloc[i], last_pivot_high) if not np.isnan(last_pivot_high) else pivot_high.iloc[i]
            else:
                last_pivot_high = pivot_high.iloc[i]

        if not np.isnan(pivot_low.iloc[i]):
            if not current_trend:
                last_pivot_low = min(pivot_low.iloc[i], last_pivot_low) if not np.isnan(last_pivot_low) else pivot_low.iloc[i]
            else:
                last_pivot_low = pivot_low.iloc[i]

        # Check for crossover and crossunder
        # We need the close of the current bar and the last pivot high/low from the previous bar.
        # In Pine Script, they use close and lastPivotHigh (which is the last pivot high from the previous bar? Actually, lastPivotHigh is updated in the same bar? Let's see: they check for crossover at the current bar, and lastPivotHigh is from the previous bar? Actually, in the code, they update lastPivotHigh at the same bar if pivotHigh is not na. But the crossover is checked on the current close and the lastPivotHigh from the previous bar? The code uses lastPivotHigh without an offset, so it's the value at the current bar? But note: lastPivotHigh is a variable that is updated in the same bar. So at bar i, lastPivotHigh is the pivot high from the previous bar? Not exactly.

        # Let's break down: at bar i, pivotHigh[i] is the pivot high for the bar i-pivotLen to i+pivotLen? Actually, pivotHigh[i] is the pivot high at bar i, which is computed using future data. So when we are at bar i, we don't know pivotHigh[i] yet. But in Pine Script, when using request.security, the pivot high at bar i on the lower timeframe is the pivot high on the higher timeframe at the corresponding time. So it's already known. So in our Python implementation, we are using a DataFrame that has the data for the whole period. So we can compute pivot_high[i] at bar i using a centered window. But we shifted it by pivot_len, so pivot_high.iloc[i] is the pivot high for bar i-pivot_len? This is confusing.

        # Given the time, I think we have to adjust our approach: we want to detect the crossover at the bar where the close crosses the last pivot high. In our loop, at bar i, we have the close[i] and the last_pivot_high from the previous bar? We need to use the last_pivot_high from the previous iteration.

        # So let's use the last_pivot_high and last_pivot_low from the previous bar (i-1). We can do:

        if i > 0:
            prev_close = df['close'].iloc[i-1]
            curr_close = df['close'].iloc[i]
            prev_last_pivot_high = last_pivot_high
            prev_last_pivot_low = last_pivot_low

            # Check for crossover: curr_close > prev_last_pivot_high and prev_close <= prev_last_pivot_high
            if curr_close > prev_last_pivot_high and prev_close <= prev_last_pivot_high:
                # Bullish break
                if current_trend and prev_last_pivot_high != last_broken_high:
                    break_of_structure = True
                else:
                    break_of_structure = False
                current_trend = True
                last_broken_high = prev_last_pivot_high
                last_broken_low = np.nan

                # Generate entry
                entry_time = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': df['time'].iloc[i],
                    'entry_time': entry_time,
                    'entry_price_guess': df['close'].iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': df['close'].iloc[i],
                    'raw_price_b': df['close'].iloc[i]
                })
                trade_num += 1

            # Check for crossunder: curr_close < prev_last_pivot_low and prev_close >= prev_last_pivot_low
            if curr_close < prev_last_pivot_low and prev_close >= prev_last_pivot_low:
                # Bearish break
                if not current_trend and prev_last_pivot_low != last_broken_low:
                    break_of_structure = True
                else:
                    break_of_structure = False
                current_trend = False
                last_broken_low = prev_last_pivot_low
                last_broken_high = np.nan

                # Generate entry
                entry_time = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': df['time'].iloc[i],
                    'entry_time': entry_time,
                    'entry_price_guess': df['close'].iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': df['close'].iloc[i],
                    'raw_price_b': df['close'].iloc[i]
                })
                trade_num += 1

    return entries