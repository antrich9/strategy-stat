import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    entries = []
    trade_num = 1

    df = df.copy()
    df['hour'] = df['time'].apply(lambda x: datetime.fromtimestamp(x/1000, tz=timezone.utc).hour)
    is_within_time = ((df['hour'] >= 8) & (df['hour'] < 10)) | ((df['hour'] >= 14) & (df['hour'] < 17))

    n = len(df)
    if n < 3:
        return entries

    bull_fvgs = []
    bear_fvgs = []

    for i in range(2, n - 1):
        if not is_within_time.iloc[i]:
            bull_fvg_to_remove = []
            bear_fvg_to_remove = []

            for j, (fvg_top, fvg_bottom) in enumerate(bull_fvgs):
                if df['low'].iloc[i] < fvg_top and df['low'].iloc[i] > fvg_bottom:
                    ts = int(df['time'].iloc[i])
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': ts,
                        'entry_time': datetime.fromtimestamp(ts/1000, tz=timezone.utc).isoformat(),
                        'entry_price_guess': float(fvg_top),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(fvg_top),
                        'raw_price_b': float(fvg_top)
                    })
                    trade_num += 1
                    bull_fvg_to_remove.append(j)

            for j, (fvg_top, fvg_bottom) in enumerate(bear_fvgs):
                if df['high'].iloc[i] > fvg_bottom and df['high'].iloc[i] < fvg_top:
                    ts = int(df['time'].iloc[i])
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': ts,
                        'entry_time': datetime.fromtimestamp(ts/1000, tz=timezone.utc).isoformat(),
                        'entry_price_guess': float(fvg_bottom),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(fvg_bottom),
                        'raw_price_b': float(fvg_bottom)
                    })
                    trade_num += 1
                    bear_fvg_to_remove.append(j)

            for idx in reversed(bull_fvg_to_remove):
                bull_fvgs.pop(idx)
            for idx in reversed(bear_fvg_to_remove):
                bear_fvgs.pop(idx)

            bull_fvg_to_remove = []
            bear_fvg_to_remove = []

            for j, (fvg_top, fvg_bottom) in enumerate(bull_fvgs):
                if df['low'].iloc[i] < fvg_bottom:
                    bull_fvg_to_remove.append(j)

            for j, (fvg_top, fvg_bottom) in enumerate(bear_fvgs):
                if df['high'].iloc[i] > fvg_top:
                    bear_fvg_to_remove.append(j)

            for idx in reversed(bull_fvg_to_remove):
                bull_fvgs.pop(idx)
            for idx in reversed(bear_fvg_to_remove):
                bear_fvgs.pop(idx)

            bull_fvg_to_remove = []
            bear_fvg_to_remove = []

            if df['low'].iloc[i - 2] >= df['high'].iloc[i]:
                bull_fvgs.append((df['low'].iloc[i], df['high'].iloc[i - 2]))

            if df['low'].iloc[i] >= df['high'].iloc[i - 2]:
                bear_fvgs.append((df['high'].iloc[i - 2], df['low'].iloc[i]))

    return entries