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

    last_ltf_high = np.nan
    last_ltf_low = np.nan

    for i in range(len(df)):
        if pd.isna(df['htf_high'].iloc[i]) or pd.isna(df['htf_low'].iloc[i]) or pd.isna(df['htf_close'].iloc[i]):
            last_ltf_high = df['ltf_high'].iloc[i]
            last_ltf_low = df['ltf_low'].iloc[i]
            continue

        htf_liq_purge_bull = (df['htf_low'].iloc[i-1] if i > 0 else np.nan) == (df['htf_low'].rolling(2).min().iloc[i] if i >= 1 else np.nan)
        htf_liq_purge_bull = not pd.isna(htf_liq_purge_bull) and df['htf_low'].iloc[i-1] == df['htf_low'].iloc[i] and df['htf_low'].iloc[i-1] <= df['htf_low'].iloc[i] and df['htf_close'].iloc[i] > df['htf_low'].iloc[i-1]

        htf_liq_purge_bear = (df['htf_high'].iloc[i-1] if i > 0 else np.nan) == (df['htf_high'].rolling(2).max().iloc[i] if i >= 1 else np.nan)
        htf_liq_purge_bear = not pd.isna(htf_liq_purge_bear) and df['htf_high'].iloc[i-1] == df['htf_high'].iloc[i] and df['htf_high'].iloc[i-1] >= df['htf_high'].iloc[i] and df['htf_close'].iloc[i] < df['htf_high'].iloc[i-1]

        ltf_mss_bull = (df['ltf_high'].iloc[i] > last_ltf_high) and (df['ltf_low'].iloc[i] > last_ltf_low)
        ltf_mss_bear = (df['ltf_low'].iloc[i] < last_ltf_low) and (df['ltf_high'].iloc[i] < last_ltf_high)

        last_ltf_high = df['ltf_high'].iloc[i]
        last_ltf_low = df['ltf_low'].iloc[i]

        long_entry = htf_liq_purge_bull and ltf_mss_bull
        short_entry = htf_liq_purge_bear and ltf_mss_bear

        if long_entry:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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

        if short_entry:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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