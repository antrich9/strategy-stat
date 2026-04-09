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
    results = []
    trade_num = 0
    n = len(df)
    PP = 5
    atrLength = 55

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    pivothigh = np.zeros(n, dtype=bool)
    pivotlow = np.zeros(n, dtype=bool)
    ph_val = np.zeros(n)
    pl_val = np.zeros(n)
    ph_idx = np.zeros(n, dtype=int)
    pl_idx = np.zeros(n, dtype=int)

    for i in range(PP, n):
        window_high = high[i-PP:i+1]
        window_low = low[i-PP:i+1]
        if high[i] == np.max(window_high):
            pivothigh[i] = True
            ph_val[i] = high[i]
            ph_idx[i] = i
        if low[i] == np.min(window_low):
            pivotlow[i] = True
            pl_val[i] = low[i]
            pl_idx[i] = i

    array_type = []
    array_value = []
    array_index = []

    bulltap = 0
    beartap = 0

    Major_HighLevel = np.nan
    Major_LowLevel = np.nan
    Major_HighIndex = -1
    Major_LowIndex = -1
    Major_HighType = ""
    Major_LowType = ""

    Bullish_Major_BoS = False
    Bearish_Major_BoS = False
    Bullish_Minor_BoS = False
    Bearish_Minor_BoS = False

    dbTradeTriggered = False
    dtTradeTriggered = False

    for i in range(PP, n):
        if pivothigh[i] or pivotlow[i]:
            if len(array_type) == 0:
                if pivothigh[i]:
                    array_type.append("H")
                    array_value.append(ph_val[i])
                    array_index.append(ph_idx[i])
                else:
                    array_type.append("L")
                    array_value.append(pl_val[i])
                    array_index.append(pl_idx[i])
            elif len(array_type) >= 1:
                if pivothigh[i]:
                    if array_type[-1] == "L" or array_type[-1] == "LL":
                        if ph_val[i] > array_value[-1]:
                            array_type.pop()
                            array_value.pop()
                            array_index.pop()
                            if len(array_type) > 1 and array_value[-1] < pl_val[i]:
                                array_type.append("HL")
                            else:
                                array_type.append("H")
                            array_value.append(ph_val[i])
                            array_index.append(ph_idx[i])
                    elif array_type[-1] == "H" or array_type[-1] == "HH":
                        if len(array_type) > 1 and array_value[-2] > pl_val[i]:
                            array_type[-1] = "HH"
                        else:
                            array_type[-1] = "H"
                        array_value[-1] = ph_val[i]
                        array_index[-1] = ph_idx[i]
                    Major_HighLevel = ph_val[i]
                    Major_HighIndex = ph_idx[i]
                    Major_HighType = array_type[-1]
                    bulltap = ph_idx[i]
                    beartap = 0
                    if len(array_type) > 1:
                        if array_type[-1] == "HH":
                            Bearish_Major_BoS = False
                        elif array_type[-1] == "HL":
                            Bearish_Major_BoS = False
                        if array_type[-2] == "HH" and array_type[-1] == "HL":
                            Bearish_Major_BoS = True
                        elif array_type[-2] == "H" and array_type[-1] == "HL":
                            Bearish_Major_BoS = True
                elif pivotlow[i]:
                    if array_type[-1] == "H" or array_type[-1] == "HH":
                        if pl_val[i] < array_value[-1]:
                            array_type.pop()
                            array_value.pop()
                            array_index.pop()
                            if len(array_type) > 1 and array_value[-1] > ph_val[i]:
                                array_type.append("LH")
                            else:
                                array_type.append("L")
                            array_value.append(pl_val[i])
                            array_index.append(pl_idx[i])
                    elif array_type[-1] == "L" or array_type[-1] == "LL":
                        if len(array_type) > 1 and array_value[-2] < ph_val[i]:
                            array_type[-1] = "LL"
                        else:
                            array_type[-1] = "L"
                        array_value[-1] = pl_val[i]
                        array_index[-1] = pl_idx[i]
                    Major_LowLevel = pl_val[i]
                    Major_LowIndex = pl_idx[i]
                    Major_LowType = array_type[-1]
                    beartap = pl_idx[i]
                    bulltap = 0
                    if len(array_type) > 1:
                        if array_type[-1] == "LL":
                            Bullish_Major_BoS = False
                        elif array_type[-1] == "LH":
                            Bullish_Major_BoS = False
                        if array_type[-2] == "LL" and array_type[-1] == "LH":
                            Bullish_Major_BoS = True
                        elif array_type[-2] == "L" and array_type[-1] == "LH":
                            Bullish_Major_BoS = True

        if i > 0:
            if close[i] > Major_LowLevel and not np.isnan(Major_LowLevel):
                if Bullish_Major_BoS:
                    dtTradeTriggered = True
            if close[i] < Major_HighLevel and not np.isnan(Major_HighLevel):
                if Bearish_Major_BoS:
                    dbTradeTriggered = True

        if dtTradeTriggered:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(close[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            dtTradeTriggered = False

        if dbTradeTriggered:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(close[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            dbTradeTriggered = False

    return results