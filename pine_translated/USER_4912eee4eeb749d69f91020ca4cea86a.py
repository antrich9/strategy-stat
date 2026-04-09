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
    atr_length = 14
    atr_multiplier = 1.0
    lookback_period = 5
    violation_threshold = 0.0001
    min_rr = 1.0

    entries = []
    trade_num = 0

    atr_vals = np.full(len(df), np.nan)
    tr = np.maximum(df['high'] - df['low'], 
                    np.maximum(np.abs(df['high'] - df['close'].shift(1)),
                               np.abs(df['close'].shift(1) - df['low'])))
    atr_vals[atr_length] = tr[:atr_length+1].mean()
    for i in range(atr_length + 1, len(df)):
        atr_vals[i] = (atr_vals[i-1] * (atr_length - 1) + tr.iloc[i]) / atr_length

    df = df.copy()
    df['atr'] = atr_vals

    lower_low = df['low'] < df['low'].shift(1)
    lower_close = df['close'] < df['close'].shift(1)
    lower_low_lower_close = lower_low & lower_close

    higher_high = df['high'] > df['high'].shift(1)
    higher_close = df['close'] > df['close'].shift(1)
    higher_high_higher_close = higher_high & higher_close

    new_structure_low = np.nan
    new_structure_high = np.nan
    prev_structure_low = np.nan
    prev_structure_high = np.nan
    bearish_trend = False
    bullish_trend = False

    entry1_zone = np.nan
    entry1_zone_top = np.nan
    entry1_zone_bottom = np.nan

    double_top_high = np.nan
    double_top_formed = False
    last_double_top_bar = -1

    double_bottom_low = np.nan
    double_bottom_formed = False
    last_double_bottom_bar = -1

    for i in range(1, len(df)):
        if np.isnan(df['atr'].iloc[i]):
            continue

        if lower_low_lower_close.iloc[i]:
            new_structure_low = df['low'].iloc[i]
            prev_structure_low = df['low'].iloc[i-1]
            bearish_trend = True
            bullish_trend = False

        if higher_high_higher_close.iloc[i]:
            new_structure_high = df['high'].iloc[i]
            prev_structure_high = df['high'].iloc[i-1]
            bullish_trend = True
            bearish_trend = False

        if bearish_trend and not np.isnan(prev_structure_low):
            entry1_zone = prev_structure_low
            entry1_zone_top = prev_structure_low + (df['atr'].iloc[i] * 0.1)
            entry1_zone_bottom = prev_structure_low - (df['atr'].iloc[i] * 0.1)

        if bullish_trend and not np.isnan(prev_structure_high):
            entry1_zone = prev_structure_high
            entry1_zone_top = prev_structure_high + (df['atr'].iloc[i] * 0.1)
            entry1_zone_bottom = prev_structure_high - (df['atr'].iloc[i] * 0.1)

        in_entry1_zone = False
        if bearish_trend and not np.isnan(entry1_zone):
            in_entry1_zone = df['high'].iloc[i] >= entry1_zone_bottom and df['low'].iloc[i] <= entry1_zone_top
        if bullish_trend and not np.isnan(entry1_zone):
            in_entry1_zone = df['high'].iloc[i] >= entry1_zone_bottom and df['low'].iloc[i] <= entry1_zone_top

        if bullish_trend and in_entry1_zone:
            if df['low'].iloc[i] >= entry1_zone_bottom and df['low'].iloc[i] <= entry1_zone_top:
                double_bottom_low = df['low'].iloc[i]
                last_double_bottom_bar = i

            if not np.isnan(double_bottom_low) and (i - last_double_bottom_bar) <= lookback_period:
                if abs(df['low'].iloc[i] - double_bottom_low) <= (double_bottom_low * violation_threshold):
                    double_bottom_formed = True
                elif df['low'].iloc[i] < double_bottom_low:
                    double_bottom_low = df['low'].iloc[i]
                    last_double_bottom_bar = i
                    double_bottom_formed = False

        if bearish_trend and in_entry1_zone:
            if df['high'].iloc[i] >= entry1_zone_bottom and df['high'].iloc[i] <= entry1_zone_top:
                double_top_high = df['high'].iloc[i]
                last_double_top_bar = i

            if not np.isnan(double_top_high) and (i - last_double_top_bar) <= lookback_period:
                if abs(df['high'].iloc[i] - double_top_high) <= (double_top_high * violation_threshold):
                    double_top_formed = True
                elif df['high'].iloc[i] > double_top_high:
                    double_top_high = df['high'].iloc[i]
                    last_double_top_bar = i
                    double_top_formed = False

        target_level = np.nan
        if bearish_trend and not np.isnan(new_structure_low):
            target_level = new_structure_low
        if bullish_trend and not np.isnan(new_structure_high):
            target_level = new_structure_high

        stop_level = np.nan
        if bearish_trend and double_top_formed and not np.isnan(double_top_high):
            stop_level = double_top_high + (df['atr'].iloc[i] * atr_multiplier)
        if bullish_trend and double_bottom_formed and not np.isnan(double_bottom_low):
            stop_level = double_bottom_low - (df['atr'].iloc[i] * atr_multiplier)

        long_condition = bullish_trend and double_bottom_formed and not np.isnan(stop_level) and not np.isnan(target_level)
        if long_condition:
            risk = abs(df['close'].iloc[i] - stop_level)
            reward = abs(target_level - df['close'].iloc[i])
            rr = reward / risk if risk != 0 else 0
            long_condition = long_condition and (rr >= min_rr)

        short_condition = bearish_trend and double_top_formed and not np.isnan(stop_level) and not np.isnan(target_level)
        if short_condition:
            risk = abs(df['close'].iloc[i] - stop_level)
            reward = abs(target_level - df['close'].iloc[i])
            rr = reward / risk if risk != 0 else 0
            short_condition = short_condition and (rr >= min_rr)

        if long_condition:
            trade_num += 1
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            double_bottom_formed = False
            double_bottom_low = np.nan

        if short_condition:
            trade_num += 1
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            double_top_formed = False
            double_top_high = np.nan

    return entries