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
    # Default parameters
    pivot_len = 25
    wbr = 0.5
    cooldown = 3
    max_lookback = 500
    trade_small = True
    trade_medium = True
    trade_large = True
    entry_mode = "Sweep+FVG"
    entry_method = "FVG"
    fvg_type = "Inverse"
    fvg_mode = "Aggressive"
    entry_expiry = 10
    trade_longs = True
    trade_shorts = True
    use_session = False
    use_htf_trend = False

    n = len(df)
    if n < 3:
        return []

    close = df['close']
    open_arr = df['open']
    high = df['high']
    low = df['low']
    times = df['time']

    # Wilder ATR calculation
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()

    # Calculate pivots
    ph = pd.Series(np.nan, index=df.index)
    pl = pd.Series(np.nan, index=df.index)
    for i in range(pivot_len, n - pivot_len):
        if i >= pivot_len and i < n - pivot_len:
            if all(high.iloc[i - pivot_len:i] <= high.iloc[i]):
                ph.iloc[i] = high.iloc[i]
            if all(low.iloc[i - pivot_len:i] >= low.iloc[i]):
                pl.iloc[i] = low.iloc[i]

    # Rolling window of last 5 pivots
    buy_side_liqs = []
    buy_side_bars = []
    sell_side_liqs = []
    sell_side_bars = []

    last_buy_grab = -999
    last_sell_grab = -999

    armed_long_setup = pd.Series(False, index=df.index)
    armed_short_setup = pd.Series(False, index=df.index)
    armed_long_bar = pd.Series(-999, index=df.index, dtype=float)
    armed_short_bar = pd.Series(-999, index=df.index, dtype=float)

    entries = []
    trade_num = 1

    for i in range(1, n):
        bar_idx = i
        last_bar = n - 1

        # Update liquidity arrays with current pivots
        if not pd.isna(ph.iloc[i]):
            liq_val = ph.iloc[i]
            liq_bar = i - pivot_len
            buy_side_liqs.append((liq_val, liq_bar))
            if len(buy_side_liqs) > 5:
                buy_side_liqs.pop(0)
                buy_side_bars.pop(0)
            buy_side_bars.append(liq_bar)

        if not pd.isna(pl.iloc[i]):
            liq_val = pl.iloc[i]
            liq_bar = i - pivot_len
            sell_side_liqs.append((liq_val, liq_bar))
            if len(sell_side_liqs) > 5:
                sell_side_liqs.pop(0)
                sell_side_bars.pop(0)
            sell_side_bars.append(liq_bar)

        # Lookback check
        in_lookback = (last_bar - bar_idx) < max_lookback
        if not in_lookback:
            continue

        grab_found = False
        grab_buyside = True
        grab_size = 0
        grab_liq_val = 0.0

        body_size = abs(close.iloc[i] - open_arr.iloc[i])

        # Check sell side for bullish grab
        if not grab_found and sell_side_liqs:
            for j in range(len(sell_side_liqs) - 1, -1, -1):
                cur_liq = sell_side_liqs[j][0]
                invalidate_bot = min(close.iloc[i], open_arr.iloc[i])
                invalidate_bot_close = invalidate_bot

                if invalidate_bot_close < cur_liq and invalidate_bot > cur_liq:
                    wick_size = invalidate_bot - low.iloc[i]
                    cur_wbr = body_size / wick_size if body_size > 0 else 0.0
                    grab_size_raw = min(cur_wbr / wbr, 3.0)
                    if grab_size_raw >= 1:
                        grab_found = True
                        grab_buyside = False
                        grab_size = int(grab_size_raw)
                        grab_liq_val = cur_liq
                        sell_side_liqs.pop(j)
                        sell_side_bars.pop(j)
                        break

                if invalidate_bot_close > cur_liq:
                    sell_side_liqs.pop(j)
                    sell_side_bars.pop(j)
                    break

        # Check buy side for bearish grab
        if not grab_found and buy_side_liqs:
            for j in range(len(buy_side_liqs) - 1, -1, -1):
                cur_liq = buy_side_liqs[j][0]
                invalidate_top = max(close.iloc[i], open_arr.iloc[i])
                invalidate_top_close = invalidate_top

                if invalidate_top_close > cur_liq and invalidate_top > cur_liq:
                    wick_size = high.iloc[i] - invalidate_top
                    cur_wbr = body_size / wick_size if body_size > 0 else 0.0
                    grab_size_raw = min(cur_wbr / wbr, 3.0)
                    if grab_size_raw >= 1:
                        grab_found = True
                        grab_buyside = True
                        grab_size = int(grab_size_raw)
                        grab_liq_val = cur_liq
                        buy_side_liqs.pop(j)
                        buy_side_bars.pop(j)
                        break

                if invalidate_top_close > cur_liq:
                    buy_side_liqs.pop(j)
                    buy_side_bars.pop(j)
                    break

        # Size filter
        size_ok = (grab_size == 1 and trade_small) or (grab_size == 2 and trade_medium) or (grab_size >= 3 and trade_large)

        # Cooldown check
        cooldown_ok = True
        if grab_buyside and grab_found:
            cooldown_ok = (i - last_buy_grab) > cooldown
        elif not grab_buyside and grab_found:
            cooldown_ok = (i - last_sell_grab) > cooldown

        # Arm setup for Sweep+FVG mode
        if grab_found and size_ok and cooldown_ok:
            if grab_buyside and trade_longs:
                if entry_mode == "Sweep+FVG":
                    armed_long_setup.iloc[i] = True
                    armed_long_bar.iloc[i] = i
                last_buy_grab = i
            elif not grab_buyside and trade_shorts:
                if entry_mode == "Sweep+FVG":
                    armed_short_setup.iloc[i] = True
                    armed_short_bar.iloc[i] = i
                last_sell_grab = i

        # FVG detection
        bullish_fvg = False
        bearish_fvg = False
        if i >= 2 and fvg_type == "Inverse":
            gap_down = low.iloc[i] > high.iloc[i - 1]
            gap_up = high.iloc[i] < low.iloc[i - 1]
            if fvg_mode == "Aggressive":
                if entry_method in ["FVG", "FVG then Engulfing"]:
                    bullish_fvg = gap_down and close.iloc[i] > low.iloc[i]
                    bearish_fvg = gap_up and close.iloc[i] < high.iloc[i]
            elif fvg_mode == "Conservative":
                if entry_method in ["FVG", "FVG then Engulfing"]:
                    prev_mid = (open_arr.iloc[i - 1] + close.iloc[i - 1]) / 2
                    bullish_fvg = gap_down and close.iloc[i] > prev_mid
                    bearish_fvg = gap_up and close.iloc[i] < prev_mid

        # Engulfing detection
        bullish_engulfing = False
        bearish_engulfing = False
        if i >= 1 and entry_method in ["Engulfing", "FVG then Engulfing"]:
            prev_bull = close.iloc[i - 1] > open_arr.iloc[i - 1]
            prev_bear = close.iloc[i - 1] < open_arr.iloc[i - 1]
            curr_bull = close.iloc[i] > open_arr.iloc[i]
            curr_bear = close.iloc[i] < open_arr.iloc[i]
            bullish_engulfing = prev_bear and curr_bull and close.iloc[i] > open_arr.iloc[i - 1] and open_arr.iloc[i] < close.iloc[i - 1]
            bearish_engulfing = prev_bull and curr_bear and close.iloc[i] < open_arr.iloc[i - 1] and open_arr.iloc[i] > close.iloc[i - 1]

        # Entry conditions
        should_entry_long = False
        should_entry_short = False
        entry_price = 0.0

        if entry_mode == "Direct":
            if grab_found and size_ok and cooldown_ok:
                if grab_buyside and trade_longs:
                    should_entry_long = True
                    entry_price = close.iloc[i]
                    last_buy_grab = i
                elif not grab_buyside and trade_shorts:
                    should_entry_short = True
                    entry_price = close.iloc[i]
                    last_sell_grab = i
        elif entry_mode == "Sweep+FVG":
            # Check if any armed long setup is still valid
            for k in range(max(0, i - entry_expiry), i):
                if armed_long_setup.iloc[k] and (i - k) <= entry_expiry:
                    if fvg_type == "Inverse":
                        if bullish_fvg or (entry_method in ["Engulfing", "FVG then Engulfing"] and bullish_engulfing):
                            should_entry_long = True
                            entry_price = close.iloc[i]
                            armed_long_setup.iloc[k] = False
                            break
            for k in range(max(0, i - entry_expiry), i):
                if armed_short_setup.iloc[k] and (i - k) <= entry_expiry:
                    if fvg_type == "Inverse":
                        if bearish_fvg or (entry_method in ["Engulfing", "FVG then Engulfing"] and bearish_engulfing):
                            should_entry_short = True
                            entry_price = close.iloc[i]
                            armed_short_setup.iloc[k] = False
                            break

        # Generate entries
        if should_entry_long:
            ts = int(times.iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif should_entry_short:
            ts = int(times.iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries