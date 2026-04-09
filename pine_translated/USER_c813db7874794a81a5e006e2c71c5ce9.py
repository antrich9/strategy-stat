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

    # Wilder ATR(144) and ATR(14)
    def wilder_atr(data_high, data_low, data_close, period):
        tr = pd.DataFrame({'h_l': data_high - data_low,
                           'h_c': np.abs(data_high - data_close.shift(1)),
                           'l_c': np.abs(data_low - data_close.shift(1))}).max(axis=1)
        atr = pd.Series(index=tr.index, dtype=float)
        atr.iloc[period - 1] = tr.iloc[:period].mean()
        multiplier = 1.0 / period
        for i in range(period, len(tr)):
            atr.iloc[i] = atr.iloc[i - 1] * (1 - multiplier) + tr.iloc[i] * multiplier
        return atr

    atr144 = wilder_atr(high, low, close, 144)

    # FVG detection helpers
    # bullG = low > high[1]
    bullG = low > high.shift(1)
    # bearG = high < low[1]
    bearG = high < low.shift(1)

    # fvgTH = 0.5 from input (default), so atr = atr144 * 0.5
    atr_filter = atr144 * 0.5

    # Bullish FVG detection
    # bull = (low - high[2]) > atr_filter AND low > high[2] AND close[1] > high[2] AND NOT (bullG OR bullG[1])
    bull_cond = (
        (low - high.shift(2)) > atr_filter
    ) & (
        low > high.shift(2)
    ) & (
        close.shift(1) > high.shift(2)
    ) & (
        ~bullG
    ) & (
        ~bullG.shift(1).fillna(False)
    )

    # Bearish FVG detection
    # bear = (low[2] - high) > atr_filter AND high < low[2] AND close[1] < low[2] AND NOT (bearG OR bearG[1])
    bear_cond = (
        (low.shift(2) - high) > atr_filter
    ) & (
        high < low.shift(2)
    ) & (
        close.shift(1) < low.shift(2)
    ) & (
        ~bearG
    ) & (
        ~bearG.shift(1).fillna(False)
    )

    # Initialize state variables for FVG tracking
    bullFvgUpper = pd.Series(np.nan, index=df.index)
    bullFvgLower = pd.Series(np.nan, index=df.index)
    bullMidpoint = pd.Series(np.nan, index=df.index)
    bullstop = pd.Series(np.nan, index=df.index)
    last_bull_fvg = pd.Series(False, index=df.index)

    bearFvgUpper = pd.Series(np.nan, index=df.index)
    bearFvgLower = pd.Series(np.nan, index=df.index)
    bearMidpoint = pd.Series(np.nan, index=df.index)
    bearstop = pd.Series(np.nan, index=df.index)
    last_bear_fvg = pd.Series(False, index=df.index)

    has_fvg = pd.Series(False, index=df.index)
    last_pct = pd.Series(np.nan, index=df.index)
    last_direction = pd.Series(False, index=df.index)  # True = bull, False = bear

    # Process bars
    for i in df.index:
        if i < 2:
            continue

        # Check for bullish FVG at this bar
        if bull_cond.iloc[i]:
            bullFvgUpper.iloc[i] = high.shift(2).iloc[i]
            bullFvgLower.iloc[i] = low.iloc[i]
            bullMidpoint.iloc[i] = (bullFvgUpper.iloc[i] + bullFvgLower.iloc[i]) / 2
            bullstop.iloc[i] = low.shift(2).iloc[i]
            # Clear previous FVG
            has_fvg.iloc[i] = True
            last_direction.iloc[i] = True
            last_pct.iloc[i] = np.nan  # Will be updated in next bars
            last_bull_fvg.iloc[i] = True
            last_bear_fvg.iloc[i] = False

        # Check for bearish FVG at this bar
        if bear_cond.iloc[i]:
            bearFvgUpper.iloc[i] = high.iloc[i]
            bearFvgLower.iloc[i] = low.shift(2).iloc[i]
            bearMidpoint.iloc[i] = (bearFvgUpper.iloc[i] + bearFvgLower.iloc[i]) / 2
            bearstop.iloc[i] = high.shift(2).iloc[i]
            has_fvg.iloc[i] = True
            last_direction.iloc[i] = False
            last_pct.iloc[i] = np.nan
            last_bull_fvg.iloc[i] = False
            last_bear_fvg.iloc[i] = True

    # Simulate FVG update with mitigation percentage
    fvg_active = False
    active_upper = np.nan
    active_lower = np.nan
    active_midpoint = np.nan
    active_is_bull = False
    active_pct = 0.0

    for i in df.index:
        if i < 2:
            continue

        # Detect new FVG
        if bull_cond.iloc[i]:
            fvg_active = True
            active_upper = high.shift(2).iloc[i]
            active_lower = low.iloc[i]
            active_midpoint = (active_upper + active_lower) / 2
            active_is_bull = True
            active_pct = 0.0
        elif bear_cond.iloc[i]:
            fvg_active = True
            active_upper = high.iloc[i]
            active_lower = low.shift(2).iloc[i]
            active_midpoint = (active_upper + active_lower) / 2
            active_is_bull = False
            active_pct = 0.0

        # Update mitigation percentage if FVG is active
        if fvg_active:
            current_high = high.iloc[i]
            current_low = low.iloc[i]

            if active_is_bull:
                top_mid = active_midpoint
                bot_mid = active_lower
                top_upper = active_upper
                bot_upper = active_lower
                tot_range = active_upper - active_lower
                if tot_range > 0:
                    prev_pct = active_pct
                    if current_low <= active_upper and current_high >= active_lower:
                        if current_low > active_lower:
                            new_bot = min(current_low, active_lower)
                            active_pct = (top_mid - new_bot) / (top_mid - bot_upper)
                        else:
                            active_pct = 1.0
                            fvg_active = False
                    last_pct.iloc[i] = active_pct
                    last_direction.iloc[i] = True
                    has_fvg.iloc[i] = True
            else:
                top_mid = active_upper
                bot_mid = active_midpoint
                top_upper = active_upper
                bot_upper = active_lower
                tot_range = active_upper - active_lower
                if tot_range > 0:
                    if current_low <= active_upper and current_high >= active_lower:
                        if current_high < active_upper:
                            new_top = max(current_high, active_upper)
                            active_pct = (new_top - bot_mid) / (top_upper - bot_mid)
                        else:
                            active_pct = 1.0
                            fvg_active = False
                    last_pct.iloc[i] = active_pct
                    last_direction.iloc[i] = False
                    has_fvg.iloc[i] = True

    # Entry conditions
    # Condition: strategy.position_size == 0 (no position), fvg_active (has FVG), last_pct > 0.01 and <= 1.0
    # Then:
    #   If last_direction == True (bull) AND crossunder(low, bullMidpoint) -> LONG
    #   If last_direction == False (bear) AND crossover(high, bearMidpoint) -> SHORT

    entries = []
    trade_num = 1
    position_open = False

    for i in df.index:
        if i < 2:
            continue

        if position_open:
            continue

        cond_global = (
            not position_open
            and has_fvg.iloc[i]
            and pd.notna(last_pct.iloc[i])
            and last_pct.iloc[i] > 0.01
            and last_pct.iloc[i] <= 1.0
        )

        if not cond_global:
            continue

        is_bull = last_direction.iloc[i]

        if is_bull:
            # crossunder(low, bullMidpoint)
            bull_mid_val = (bullFvgUpper.iloc[i] + bullFvgLower.iloc[i]) / 2 if pd.notna(bullFvgUpper.iloc[i]) and pd.notna(bullFvgLower.iloc[i]) else np.nan
            if pd.notna(bull_mid_val):
                if i > 0:
                    crossunder_cond = (low.iloc[i] < bull_mid_val) and (low.iloc[i - 1] >= bull_mid_val)
                else:
                    crossunder_cond = False
            else:
                crossunder_cond = False

            if crossunder_cond:
                entry_price = bull_mid_val
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(time.iloc[i]),
                    'entry_time': datetime.fromtimestamp(time.iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
                position_open = True
        else:
            # crossover(high, bearMidpoint)
            bear_mid_val = (bearFvgUpper.iloc[i] + bearFvgLower.iloc[i]) / 2 if pd.notna(bearFvgUpper.iloc[i]) and pd.notna(bearFvgLower.iloc[i]) else np.nan
            if pd.notna(bear_mid_val):
                if i > 0:
                    crossover_cond = (high.iloc[i] > bear_mid_val) and (high.iloc[i - 1] <= bear_mid_val)
                else:
                    crossover_cond = False
            else:
                crossover_cond = False

            if crossover_cond:
                entry_price = bear_mid_val
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(time.iloc[i]),
                    'entry_time': datetime.fromtimestamp(time.iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
                position_open = True

    return entries