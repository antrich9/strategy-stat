import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Hardcoded parameters (default from Pine Script)
    rrTarget = 2.0
    atrMultiplier = 1.5
    atrLength = 14
    minBodyPct = 0.3
    trendFilter = "Structure Only"  # or "Structure + MA"
    maLength = 50
    fibEntryHigh = 0.786
    fibEntryLow = 0.382
    PP = 5

    # Ensure required columns
    # df columns: time, open, high, low, close, volume

    # Compute ATR (Wilder)
    # True range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    # ATR = Wilder average of TR
    # Use ewm with alpha = 1/atrLength

    # Compute EMA (if needed)
    # Use close.ewm(span=maLength, adjust=False).mean()

    # Compute pivot highs and lows
    # Use rolling max/min with window PP? But need to consider future bars.
    # For simplicity, we can compute pivots using a loop.

    # ZigZag logic: maintain arrays of swing types, values, indices.
    # We'll use lists.

    # Then generate entries based on conditions.

    # Return list of dicts.