import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    high = df['high']
    low = df['low']
    close = df['close']

    lookback_period = 5
    violation_threshold = 0.0001
    atr_length = 14
    atr_multiplier = 1.0
    target_front_run = 2

    lower_low = low < low.shift(1)
    lower_close = close < close.shift(1)
    lower_low_lower_close = lower_low & lower_close

    higher_high = high > high.shift(1)
    higher_close = close > close.shift(1)
    higher_high_higher_close = higher_high & higher_close

    n = len(df)
    bullish_trend = pd.Series(False, index=df.index)
    bearish_trend = pd.Series(False, index=df.index)
    new_structure_low = pd.Series(np.nan, index=df.index)
    new_structure_high = pd.Series(np.nan, index=df.index)
    outside_return_high = pd.Series(np.nan, index=df.index)
    outside_return_low = pd.Series(np.nan, index=df.index)
    outside_return_close = pd.Series(np.nan, index=df.index)
    entry3_zone_top = pd.Series(np.nan, index=df.index)
    entry3_zone_bottom = pd.Series(np.nan, index=df.index)
    violation_occurred = pd.Series(False, index=df.index)
    in_entry3_zone = pd.Series(False, index=df.index)
    double_top_formed = pd.Series(False, index=df.index)
    double_bottom_formed = pd.Series(False, index=df.index)
    double_top_high_arr = pd.Series(np.nan, index=df.index)
    double_bottom_low_arr = pd.Series(np.nan, index=df.index)

    last_violation_bar = -1
    last_double_top_bar = -1
    last_double_bottom_bar = -1

    for i in range(n):
        if lower_low_lower_close.iloc[i]:
            bullish_trend.iloc[i] = True
            bearish_trend.iloc[i] = False
            new_structure_low.iloc[i] = low.iloc[i]
            if i >= 2:
                outside_return_high.iloc[i] = high.iloc[i-2]
                outside_return_low.iloc[i] = low.iloc[i-2]
                outside_return_close.iloc[i] = close.iloc[i-2]

        if higher_high_higher_close.iloc[i]:
            bullish_trend.iloc[i] = False
            bearish_trend.iloc[i] = True
            new_structure_high.iloc[i] = high.iloc[i]
            if i >= 2:
                outside_return_high.iloc[i] = high.iloc[i-2]
                outside_return_low.iloc[i] = low.iloc[i-2]
                outside_return_close.iloc[i] = close.iloc[i-2]

        if i > 0:
            bullish_trend.iloc[i] = bullish_trend.iloc[i-1] or bullish_trend.iloc[i]
            bearish_trend.iloc[i] = bearish_trend.iloc[i-1] or bearish_trend.iloc[i]
            if pd.notna(new_structure_low.iloc[i-1]):
                new_structure_low.iloc[i] = new