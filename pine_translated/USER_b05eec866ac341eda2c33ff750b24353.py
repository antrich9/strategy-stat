import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Extract close series
    close = df['close']

    # Parameters for TC1 (short-term)
    lengthTC1 = 5
    factorTC1 = 0.7

    # Parameters for TC2 (long-term)
    lengthTC2 = 18
    factorTC2 = 0.7

    # GDTC function: triple EMA smoothing
    def gdTC(src, length, factor):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factor) - ema2 * factor

    # Compute TC1 and TC2 with triple smoothing
    TC1 = gdTC(gdTC(gdTC(close, lengthTC1, factorTC1), lengthTC1, factorTC1), lengthTC1, factorTC1)
    TC2 = gdTC(gdTC(gdTC(close, lengthTC2, factorTC2), lengthTC2, factorTC2), lengthTC2, factorTC2)

    # Previous values for crossover detection
    TC1_prev = TC1.shift(1)
    TC2_prev = TC2.shift(1)

    # Basic entry conditions
    basicLongCondition = (TC1 > TC1_prev) & (TC2 > TC2_prev) & (TC1 > TC2)
    basicShortCondition = (TC1 < TC1_prev) & (TC2 < TC2_prev) & (TC1 < TC2)

    # Apply user inputs (useTC=true, highlightMovementsTC=true)
    TCSignalsLong = basicLongCondition
    TCSignalsShort = basicShortCondition

    # Cross confirmation (crossTC=true)
    TCSignalsLongCross = TCSignalsLong & ~TCSignalsLong.shift(1).fillna(False)
    TCSignalsShortCross = TCSignalsShort & ~TCSignalsShort.shift(1).fillna(False)

    # Final signals (inverseTC=false)
    TCSignalsLongFinal = TCSignalsLongCross
    TCSignalsShortFinal = TCSignalsShortCross

    # Generate entry list
    entries = []
    trade_num = 1

    for i in range(1, len(df)):
        if TCSignalsLongFinal.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
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
            trade_num += 1
        elif TCSignalsShortFinal.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
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
            trade_num += 1

    return entries