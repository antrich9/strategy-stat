import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    entries = []
    trade_num = 1

    ig_active = False
    ig_c1_high = None
    ig_c1_low = None
    ig_c3_high = None
    ig_c3_low = None
    ig_direction = 0
    ig_start_bar = None
    ig_validation_end = None

    for i in range(len(df)):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        close = df['close'].iloc[i]
        ts = df['time'].iloc[i]

        if i >= 2:
            high_2 = df['high'].iloc[i-2]
            low_2 = df['low'].iloc[i-2]

            bullish_fvg = high_2 < low
            bearish_fvg = low_2 > high

            if bullish_fvg and not ig_active:
                ig_active = True
                ig_direction = 1
                ig_c1_high = high_2
                ig_c1_low = low_2
                ig_c3_high = high
                ig_c3_low = low
                ig_start_bar = i
                ig_validation_end = i + 4

            elif bearish_fvg and not ig_active:
                ig_active = True
                ig_direction = -1
                ig_c1_high = high_2
                ig_c1_low = low_2
                ig_c3_high = high
                ig_c3_low = low
                ig_start_bar = i
                ig_validation_end = i + 4

        if ig_active and ig_start_bar is not None and i > ig_start_bar and i <= ig_validation_end:
            if i > 0:
                prev_close = df['close'].iloc[i-1]
                if ig_direction == 1 and close < ig_c1_low and prev_close >= ig_c1_low:
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': int(ts),
                        'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                        'entry_price_guess': float(close),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(close),
                        'raw_price_b': float(close)
                    })
                    trade_num += 1
                    ig_active = False
                    ig_start_bar = None
                    ig_validation_end = None

                elif ig_direction == -1 and close > ig_c1_high and prev_close <= ig_c1_high:
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': int(ts),
                        'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                        'entry_price_guess': float(close),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(close),
                        'raw_price_b': float(close)
                    })
                    trade_num += 1
                    ig_active = False
                    ig_start_bar = None
                    ig_validation_end = None

        if ig_active and ig_validation_end is not None and i > ig_validation_end:
            ig_active = False
            ig_start_bar = None
            ig_validation_end = None

    return entries