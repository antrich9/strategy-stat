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
    # Preserve input and work on a copy
    df = df.copy()

    # ── Parameters (matching Pine Script defaults) ──────────────────────────────
    length_gpi = 14          # lengthGPI
    sma_length_gpi = 9       # sma_lengthGPI

    # ── Compute GPI (scaled sum of close-open) ──────────────────────────────────
    # Pine: gpiGPI = 100 * (math.sum(close - open, lengthGPI) / (lengthGPI * syminfo.mintick))
    # The constant factor (100 / (lengthGPI * syminfo.mintick)) is irrelevant for crossover,
    # so we use the raw rolling sum.
    df['diff'] = df['close'] - df['open']
    df['gpi'] = df['diff'].rolling(window=length_gpi).sum()
    df['gpi_sma'] = df['gpi'].rolling(window=sma_length_gpi).mean()

    # ── Generate entries ────────────────────────────────────────────────────────
    entries = []
    trade_num = 1

    for i in range(len(df)):
        gpi_val = df['gpi'].iloc[i]
        gpi_sma_val = df['gpi_sma'].iloc[i]

        # Skip bars with missing indicators
        if pd.isna(gpi_val) or pd.isna(gpi_sma_val):
            continue

        # Determine direction (useGPI=true, useInverseGPI=true → long when GPI < SMA, short when GPI > SMA)
        if gpi_val < gpi_sma_val:
            direction = 'long'
        elif gpi_val > gpi_sma_val:
            direction = 'short'
        else:
            continue

        entry_price = float(df['close'].iloc[i])
        entry_ts = int(df['time'].iloc[i])
        entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()

        entries.append({
            'trade_num': trade_num,
            'direction': direction,
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

    return entries