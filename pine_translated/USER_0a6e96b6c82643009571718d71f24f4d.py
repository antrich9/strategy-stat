import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Required columns
    time = df['time'].astype(np.int64)
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']

    # ------------------------------------------------------------------
    # 1. Bullish & Bearish Fair Value Gap detection (threshold = 0)
    # ------------------------------------------------------------------
    bull_cond1 = low > high.shift(2)
    bull_cond2 = close.shift(1) > high.shift(2)
    bull_fvg = (bull_cond1 & bull_cond2).fillna(False)

    bear_cond1 = high < low.shift(2)
    bear_cond2 = close.shift(1) < low.shift(2)
    bear_fvg = (bear_cond1 & bear_cond2).fillna(False)

    # ------------------------------------------------------------------
    # 2. Consecutive FVG counters
    # ------------------------------------------------------------------
    n = len(df)
    consec_bull = np.zeros(n, dtype=int)
    consec_bear = np.zeros(n, dtype=int)

    bull_cnt = 0
    bear_cnt = 0
    for i in range(n):
        if bull_fvg.iloc[i]:
            bull_cnt += 1
            bear_cnt = 0
        elif bear_fvg.iloc[i]:
            bear_cnt += 1
            bull_cnt = 0
        # otherwise keep previous counters
        consec_bull[i] = bull_cnt
        consec_bear[i] = bear_cnt

    # ------------------------------------------------------------------
    # 3. Swing detection (local high/low)
    # ------------------------------------------------------------------
    swing_high = (high.shift(1) < high.shift(2)) & (high.shift(3) < high.shift(2)) & (high.shift(4) < high.shift(2))
    swing_low  = (low.shift(1)  > low.shift(2))  & (low.shift(3)  > low.shift(2))  & (low.shift(4)  > low.shift(2))
    swing_high = swing_high.fillna(False)
    swing_low  = swing_low.fillna(False)

    # ------------------------------------------------------------------
    # 4. Last swing type tracking
    # ------------------------------------------------------------------
    last_swing_type = []
    cur_type = "none"
    for i in range(n):
        if swing_high.iloc[i]:
            cur_type = "high"
        elif swing_low.iloc[i]:
            cur_type = "low"
        last_swing_type.append(cur_type)

    # ------------------------------------------------------------------
    # 5. Determine bullish/bearish leg flags
    # ------------------------------------------------------------------
    is_bullish_leg = pd.Series(False, index=df.index)
    is_bearish_leg = pd.Series(False, index=df.index)

    for i in range(n):
        if consec_bull[i] == 2 and last_swing_type[i] == "low":
            is_bullish_leg.iloc[i] = True
        elif consec_bear[i] == 2 and last_swing_type[i] == "high":
            is_bearish_leg.iloc[i] = True

    # ------------------------------------------------------------------
    # 6. Build entry list
    # ------------------------------------------------------------------
    entries = []
    trade_num = 1
    for i in range(n):
        if is_bullish_leg.iloc[i] or is_bearish_leg.iloc[i]:
            direction = "long" if is_bullish_leg.iloc[i] else "short"
            entry_ts = int(time.iloc[i])
            entry_price = float(close.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
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