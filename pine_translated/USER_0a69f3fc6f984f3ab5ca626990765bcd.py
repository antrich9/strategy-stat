import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Ensure DataFrame is sorted by time (assumed)
    # Parameters
    PP = 5  # pivot period
    # Compute pivot high and low
    high = df['high']
    low = df['low']
    close = df['close']
    # We'll compute pivot high and low series using loop for clarity
    pivot_high = pd.Series(np.nan, index=df.index)
    pivot_low = pd.Series(np.nan, index=df.index)
    for i in range(PP, len(df) - PP):
        if high.iloc[i] == high.iloc[i-PP:i+PP+1].max():
            pivot_high.iloc[i] = high.iloc[i]
        if low.iloc[i] == low.iloc[i-PP:i+PP+1].min():
            pivot_low.iloc[i] = low.iloc[i]
    # Determine last swing high and low up to each bar
    # We can fill forward the last known pivot high/low
    last_swing_high = pivot_high.ffill()
    last_swing_low = pivot_low.ffill()
    # Compute entry signals
    long_signal = (close > last_swing_high) & last_swing_high.notna()
    short_signal = (close < last_swing_low) & last_swing_low.notna()
    # Build list of entries
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if long_signal.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            }
            entries.append(entry)
            trade_num += 1
        elif short_signal.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            }
            entries.append(entry)
            trade_num += 1
    return entries