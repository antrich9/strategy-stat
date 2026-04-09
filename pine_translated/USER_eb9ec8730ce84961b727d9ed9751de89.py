import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # parameters
    lookback = 30
    maxBpr = 50
    # compute series
    high = df['high']
    low = df['low']
    close = df['close']
    # compute FVGs
    bullishFVG = (low > high.shift(2)) & (close.shift(1) > high.shift(2))
    bearishFVG = (high < low.shift(2)) & (close.shift(1) < low.shift(2))
    # initialize BPR list
    bpr_list = []
    entries = []
    trade_num = 1
    n = len(df)
    # iterate bars
    for i in range(2, n):
        if bullishFVG.iloc[i]:
            p1 = high.iloc[i-2]
            p2 = low.iloc[i]
            # inner loop over offsets
            found = False
            for offset in range(2, lookback):
                j = i - offset
                if j < 0:
                    break
                # check index for low[i+2] and high[i] within range
                low_i_plus2_idx = i - (offset + 2)
                high_i_idx = j
                if low_i_plus2_idx < 0:
                    continue
                if not bearishFVG.iloc[j]:
                    continue
                if not (p1 < low.iloc[low_i_plus2_idx] and p2 > high.iloc[high_i_idx]):
                    continue
                # condition satisfied
                # add BPR to front
                bpr_list.insert(0, {})
                # overflow check
                if len(bpr_list) > maxBpr:
                    # pop oldest
                    bpr_list.pop()
                    # generate Buy entry
                    entry_price = close.iloc[i]
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
                # always generate Buy1 entry
                entry_price = close.iloc[i]
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
                found = True
                break
            if found:
                continue
        if bearishFVG.iloc[i]:
            p1 = low.iloc[i-2]
            p2 = high.iloc[i]
            found = False
            for offset in range(2, lookback):
                j = i - offset
                if j < 0:
                    break
                low_i_idx = j
                high_i_plus2_idx = i - (offset + 2)
                if high_i_plus2_idx < 0:
                    continue
                if not bullishFVG.iloc[j]:
                    continue
                if not (p1 > high.iloc[high_i_plus2_idx] and p2 < low.iloc[low_i_idx]):
                    continue
                # condition satisfied
                bpr_list.insert(0, {})
                # overflow check: if size > maxBpr, pop oldest but no entry
                if len(bpr_list) > maxBpr:
                    bpr_list.pop()
                # generate Sell entry
                entry_price = close.iloc[i]
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
                found = True
                break
    return entries