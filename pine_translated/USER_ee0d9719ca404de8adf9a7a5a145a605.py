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
    high = df['high']
    low = df['low']
    close = df['close']
    time = df['time']

    n = len(df)
    PP = 5

    # Pivot tracking arrays (size 3)
    Major_HighType_arr = [None, None, None]
    Major_HighValue_arr = [np.nan, np.nan, np.nan]
    Major_HighIndex_arr = [0, 0, 0]
    Major_LowType_arr = [None, None, None]
    Major_LowValue_arr = [np.nan, np.nan, np.nan]
    Major_LowIndex_arr = [0, 0, 0]
    Minor_HighType_arr = [None, None, None]
    Minor_HighValue_arr = [np.nan, np.nan, np.nan]
    Minor_HighIndex_arr = [0, 0, 0]
    Minor_LowType_arr = [None, None, None]
    Minor_LowValue_arr = [np.nan, np.nan, np.nan]
    Minor_LowIndex_arr = [0, 0, 0]

    dbTradeTriggered = False
    dtTradeTriggered = False
    trade_num = 0
    entries = []

    for i in range(PP, n):
        idx = i - PP

        # Detect pivots
        isHighPivot = True
        isLowPivot = True
        for j in range(1, PP + 1):
            if high.iloc[i] <= high.iloc[i - j]:
                isHighPivot = False
            if low.iloc[i] >= low.iloc[i - j]:
                isLowPivot = False

        # Update major pivot arrays on pivot detection
        if isHighPivot:
            Major_HighType_arr[2] = Major_HighType_arr[1]
            Major_HighValue_arr[2] = Major_HighValue_arr[1]
            Major_HighIndex_arr[2] = Major_HighIndex_arr[1]
            Major_HighType_arr[1] = Major_HighType_arr[0]
            Major_HighValue_arr[1] = Major_HighValue_arr[0]
            Major_HighIndex_arr[1] = Major_HighIndex_arr[0]
            Major_HighType_arr[0] = 'H'
            Major_HighValue_arr[0] = high.iloc[i]
            Major_HighIndex_arr[0] = idx

        if isLowPivot:
            Minor_LowType_arr[2] = Minor_LowType_arr[1]
            Minor_LowValue_arr[2] = Minor_LowValue_arr[1]
            Minor_LowIndex_arr[2] = Minor_LowIndex_arr[1]
            Major_LowType_arr[1] = Major_LowType_arr[0]
            Major_LowValue_arr[1] = Major_LowValue_arr[0]
            Major_LowIndex_arr[1] = Major_LowIndex_arr[0]
            Major_LowType_arr[0] = 'L'
            Major_LowValue_arr[0] = low.iloc[i]
            Major_LowIndex_arr[0] = idx

        # Check require valid pivots for entry logic
        if pd.isna(Major_HighValue_arr[0]) or pd.isna(Major_LowValue_arr[0]):
            continue
        if pd.isna(Major_HighValue_arr[1]) or pd.isna(Major_LowValue_arr[1]):
            continue

        # Calculate crossover/crossunder
        cross_above_MH0 = (close.iloc[i] > Major_HighValue_arr[0]) and (i > 0 and close.iloc[i - 1] <= Major_HighValue_arr[0])
        cross_below_ML0 = (close.iloc[i] < Major_LowValue_arr[0]) and (i > 0 and close.iloc[i - 1] >= Major_LowValue_arr[0])

        # Determine structure
        MH0 = Major_HighValue_arr[0]
        MH1 = Major_HighValue_arr[1]
        ML0 = Major_LowValue_arr[0]
        ML1 = Major_LowValue_arr[1]

        bull_ms = (MH0 > MH1) and (ML0 > ML1)
        bear_ms = (MH0 < MH1) and (ML0 < ML1)

        # Check minor structure if available
        ml_valid = not pd.isna(Minor_LowValue_arr[0]) and not pd.isna(Minor_LowValue_arr[1])
        mh_valid = not pd.isna(Minor_HighValue_arr[0]) and not pd.isna(Minor_HighValue_arr[1])

        if ml_valid:
            mL0 = Minor_LowValue_arr[0]
            mL1 = Minor_LowValue_arr[1]
            if bull_ms and mL0 <= mL1:
                bull_ms = False
            if bear_ms and mL0 >= mL1:
                bear_ms = False

        if mh_valid:
            mH0 = Minor_HighValue_arr[0]
            mH1 = Minor_HighValue_arr[1]
            if bull_ms and mH0 <= mH1:
                bull_ms = False
            if bear_ms and mH0 >= mH1:
                bear_ms = False

        # Entry logic: long on break above MH0, short on break below ML0
        # Check: not already open, structure confirmed, crossover/crossunder occurred
        if cross_above_MH0 and bull_ms and not dtTradeTriggered:
            dtTradeTriggered = True
            trade_num += 1
            entry_price = close.iloc[i]
            entry_ts = int(time.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
        elif cross_below_ML0 and bear_ms and not dbTradeTriggered:
            dbTradeTriggered = True
            trade_num += 1
            entry_price = close.iloc[i]
            entry_ts = int(time.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

    return entries