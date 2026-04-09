import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    timestamps = df['time']

    # Input parameters (defaults from Pine Script)
    pivotLookback = 5
    fib382 = 0.382
    fib618 = 0.618
    atrLen = 14
    bbLen = 20
    bbMult = 2.0

    # Session: London as default (0700-1600 UTC)
    def get_utc_hhmm(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.hour * 100 + dt.minute

    utc_hhmm = timestamps.apply(get_utc_hhmm)
    london_open = 700
    london_close = 1600
    in_session = (utc_hhmm >= london_open) & (utc_hhmm < london_close)

    # ATR calculation (Wilder)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atrLen, adjust=False).mean()
    atr_sma = atr.rolling(20).mean()
    atr_expanding = atr > atr_sma

    # Bollinger Bands
    bb_basis = close.rolling(bbLen).mean()
    bb_std = close.rolling(bbLen).std()
    bb_upper = bb_basis + bbMult * bb_std
    bb_lower = bb_basis - bbMult * bb_std
    bb_perc_b = (close - bb_lower) / (bb_upper - bb_lower)
    bb_width = (bb_upper - bb_lower) / bb_basis
    bb_width_sma = bb_width.rolling(20).mean()
    bb_squeeze = bb_width < bb_width_sma
    bb_expand_long = (~bb_squeeze) & (bb_perc_b > 0.80)
    bb_expand_short = (~bb_squeeze) & (bb_perc_b < 0.20)

    # EMA for trend bias
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    bull_bias = (close > ema50) & (ema50 > ema200)
    bear_bias = (close < ema50) & (ema50 < ema200)

    # Pivot points (Wilder style - pivotLookback bars confirmed)
    pivot_high = ta_pivothigh(high, pivotLookback)
    pivot_low = ta_pivotlow(low, pivotLookback)

    # Session times
    # Wave heights with safe denominators
    w1_long_h = np.maximum(pH1_vals - pL2_vals, 0.0001)
    w3_long_h = np.maximum(pH2_vals - pL2_vals, 0.0001)
    w1_short_h = np.maximum(pH2_vals - pL1_vals, 0.0001)
    w3_short_h = np.maximum(pH2_vals - pL2_vals, 0.0001)

    # Wave ratios
    w2_long_ratio = (pH1_vals - pL1_vals) / w1_long_h
    w4_long_ratio = (pH2_vals - pL1_vals) / w3_long_h
    w2_short_ratio = (pH1_vals - pL1_vals) / w1_short_h
    w4_short_ratio = (pH1_vals - pL2_vals) / w3_short_h

    # Wave validity flags
    wave2_long_valid = (pH1_vals.notna()) & (pL1_vals.notna()) & (pL2_vals.notna()) & \
                       (w2_long_ratio >= fib382) & (w2_long_ratio <= fib618) & \
                       (pL1_vals > pL2_vals) & (bL1_vals > bH1_vals)

    wave4_long_valid = (pH2_vals.notna()) & (pL1_vals.notna()) & (pL2_vals.notna()) & \
                       (w4_long_ratio >= fib382 * 0.5) & (w4_long_ratio <= fib382 + 0.05) & \
                       (pL1_vals > pL2_vals) & (bL1_vals > bH2_vals)

    wave2_short_valid = (pL1_vals.notna()) & (pH1_vals.notna()) & (pH2_vals.notna()) & \
                        (w2_short_ratio >= fib382) & (w2_short_ratio <= fib618) & \
                        (pH1_vals < pH2_vals) & (bH1_vals > bL1_vals)

    wave4_short_valid = (pL2_vals.notna()) & (pH1_vals.notna()) & (pH2_vals.notna()) & \
                        (w4_short_ratio >= fib382 * 0.5) & (w4_short_ratio <= fib382 + 0.05) & \
                        (pH1_vals < pH2_vals) & (bH1_vals > bL2_vals)

    # Raw wave triggers
    wave3_long_raw = wave2_long_valid & (close > pH1_vals) & bb_expand_long & atr_expanding & bull_bias
    wave5_long_raw = wave4_long_valid & (close > pH2_vals) & bb_expand_long & atr_expanding & bull_bias
    wave3_short_raw = wave2_short_valid & (close < pL1_vals) & bb_expand_short & atr_expanding & bear_bias
    wave5_short_raw = wave4_short_valid & (close < pL2_vals) & bb_expand_short & atr_expanding & bear_bias

    # Confirmed triggers (only on closed bars)
    wave3_long_trigger = wave3_long_raw & in_session
    wave5_long_trigger = wave5_long_raw & in_session
    wave3_short_trigger = wave3_short_raw & in_session
    wave5_short_trigger = wave5_short_raw & in_session

    long_triggered = wave3_long_trigger | wave5_long_trigger
    short_triggered = wave3_short_trigger | wave5_short_trigger

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        entry_price_guess = close.iloc[i]
        direction = None

        if long_triggered.iloc[i]:
            direction = 'long'
        elif short_triggered.iloc[i]:
            direction = 'short'

        if direction is not None:
            ts = int(timestamps.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1

    return entries