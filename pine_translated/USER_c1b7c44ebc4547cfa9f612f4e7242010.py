import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    body = (df['close'] - df['open']).abs()
    candle_range = df['high'] - df['low']
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    total_wick = upper_wick + lower_wick
    body_pct = body / candle_range

    wick_to_body_ratio = 0.3
    min_body_pct = 0.6

    is_respect_bull = lower_wick > body * 1.5
    is_respect_bear = upper_wick > body * 1.5
    is_disrespect_candle = (total_wick / body < wick_to_body_ratio) & (body_pct >= min_body_pct)

    swing_high_2 = df['high'].shift(2)
    swing_high_3 = df['high'].shift(3)
    swing_high_4 = df['high'].shift(4)
    swing_low_2 = df['low'].shift(2)
    swing_low_3 = df['low'].shift(3)
    swing_low_4 = df['low'].shift(4)

    is_swing_high = (swing_high_2 < df['high']) & (swing_high_3.shift(3) < swing_high_2) & (swing_high_4.shift(4) < swing_high_2)
    is_swing_low = (swing_low_2 > df['low']) & (swing_low_3.shift(3) > swing_low_2) & (swing_low_4.shift(4) > swing_low_2)

    bull_fvg = (df['low'] > swing_high_2) & (df['close'].shift(1) > swing_high_2)
    bear_fvg = (df['high'] < swing_low_2) & (df['close'].shift(1) < swing_low_2)

    threshold = 0.0

    bull_fvg = bull_fvg & ((df['low'] - swing_high_2) / swing_high_2 > threshold)
    bear_fvg = bear_fvg & ((swing_low_2 - df['high']) / df['high'] > threshold)

    bull_fvg_final = bull_fvg & is_respect_bull
    bear_fvg_final = bear_fvg & is_respect_bear

    prev_day_high_sweep = df['high'] > df['high'].shift(1)
    prev_day_low_sweep = df['low'] < df['low'].shift(1)

    entries = []

    bull_conditions = bull_fvg_final | prev_day_high_sweep
    bear_conditions = bear_fvg_final | prev_day_low_sweep

    for i in range(len(df)):
        if i < 2:
            continue

        if pd.isna(is_respect_bull.iloc[i]) or pd.isna(bull_fvg.iloc[i]):
            continue

        direction = None
        if bull_conditions.iloc[i]:
            direction = 'long'
        elif bear_conditions.iloc[i]:
            direction = 'short'

        if direction:
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': len(entries) + 1,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })

    return entries