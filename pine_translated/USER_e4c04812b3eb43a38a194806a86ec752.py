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
    # -------------------------------------------------------------------------
    # Helper: Wilder ATR (same as ta.atr(len))
    # -------------------------------------------------------------------------
    def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.ewm(alpha=1.0 / length, adjust=False).mean()
        return atr

    # -------------------------------------------------------------------------
    # Helper: SuperTrend direction (1 = bullish, -1 = bearish, 0 = unknown)
    # -------------------------------------------------------------------------
    def supertrend_direction(high: pd.Series, low: pd.Series, close: pd.Series,
                            period: int = 10, multiplier: float = 3.0) -> pd.Series:
        atr = wilder_atr(high, low, close, period)
        hl2 = (high + low) / 2.