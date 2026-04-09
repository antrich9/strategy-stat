import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Compute New York hour for session filter
    ts = pd.to_datetime(df['time'], unit='s', utc=True)
    ny_hour = ts.dt.tz_convert('America/New_York').dt.hour
    time_in_session = (ny_hour >= 10) & (ny_hour < 11)

    # Detect FVGs
    fvg_up = (
        (df['low'] > df['high'].shift(2)) &
        (df['high'].shift(1) > df['high'].shift(2))
    ).fillna(False)

    fvg_down = (
        (df['high'] < df['low'].shift(2)) &
        (df['low'].shift(1) < df['low'].shift(2))
    ).fillna(False)

    entries = []
    trade_num = 1
    fvg_triggered = False

    for i in range(len(df)):
        if not time_in_session.iloc[i]:
            fvg_triggered = False
        else:
            if not fvg_triggered:
                if fvg_up.iloc[i]:
                    entry_ts = int(df['time'].iloc[i])
                    entry_price = float(df['close'].iloc[i])
                    entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': entry_ts,
                        'entry_time': entry_time_str,
                        'entry_price_guess': entry_price,
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': entry_price,
                        'raw_price_b': entry_price
                    })
                    trade_num += 1
                    fvg_triggered = True
                elif fvg_down.iloc[i]:
                    entry_ts = int(df['time'].iloc[i])
                    entry_price = float(df['close'].iloc[i])
                    entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': entry_ts,
                        'entry_time': entry_time_str,
                        'entry_price_guess': entry_price,
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': entry_price,
                        'raw_price_b': entry_price
                    })
                    trade_num += 1
                    fvg_triggered = True

    return entries