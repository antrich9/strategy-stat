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
    close = df['close']
    high = df['high']
    low = df['low']

    # Parameters
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2

    bb = input_lookback

    # Pivot calculations
    pl = close.rolling(window=bb + 1).apply(lambda x: np.nan if pd.isna(x.iloc[0]) else x.idxmin(), raw=True)
    pl = pd.Series([np.nan] * len(df), index=df.index)
    for i in range(bb, len(df)):
        window = low.iloc[i - bb:i + 1]
        min_idx = window.idxmin()
        pl.iloc[i] = min_idx

    ph = pd.Series([np.nan] * len(df), index=df.index)
    for i in range(bb, len(df)):
        window = high.iloc[i - bb:i + 1]
        max_idx = window.idxmax()
        ph.iloc[i] = max_idx

    pl = low.rolling(window=bb + 1, min_periods=1).apply(lambda x: x.iloc[0] if len(x) > bb and pd.isna(x.iloc[:bb].min()) else np.nan, raw=True)
    ph = high.rolling(window=bb + 1, min_periods=1).apply(lambda x: x.iloc[0] if len(x) > bb and pd.isna(x.iloc[:bb].min()) else np.nan, raw=True)

    pl = pd.Series([np.nan] * len(df), index=df.index)
    ph_vals = pd.Series([np.nan] * len(df), index=df.index)
    for i in range(bb, len(df)):
        window_low = low.iloc[i - bb:i + 1].values
        window_high = high.iloc[i - bb:i + 1].values
        pl.iloc[i] = window_low[bb]
        ph_vals[i] = window_high[bb]

    pl = pd.Series([np.nan] * len(df), index=df.index)
    ph_vals = pd.Series([np.nan] * len(df), index=df.index)
    for i in range(bb, len(df)):
        local_min_idx = low.iloc[i - bb:i + 1].idxmin()
        pl.iloc[i] = low.loc[local_min_idx]
        local_max_idx = high.iloc[i - bb:i + 1].idxmax()
        ph_vals[i] = high.loc[local_max_idx]
    ph = ph_vals

    # Box boundaries
    s_y_loc = pd.Series([np.nan] * len(df), index=df.index)
    r_y_loc = pd.Series([np.nan] * len(df), index=df.index)
    for i in range(bb + 1, len(df)):
        low_bp1 = low.iloc[i - bb - 1] if i - bb - 1 >= 0 else np.nan
        low_bm1 = low.iloc[i - bb - 1] if i - bb - 1 >= 0 else np.nan
        if not pd.isna(low.iloc[i - bb]) and not pd.isna(low.iloc[i - bb - 2]):
            s_y_loc.iloc[i] = low.iloc[i - bb - 2] if low.iloc[i - bb - 1] > low.iloc[i - bb - 2] else low.iloc[i - bb - 1]
        if not pd.isna(high.iloc[i - bb]) and not pd.isna(high.iloc[i - bb - 2]):
            r_y_loc.iloc[i] = high.iloc[i - bb - 2] if high.iloc[i - bb - 1] > high.iloc[i - bb - 2] else high.iloc[i - bb - 1]

    s_bot = pl
    s_top = s_y_loc
    r_bot = r_y_loc
    r_top = ph_vals

    # ATR calculation (Wilder)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()

    # Breakout tracking
    s_break = pd.Series([False] * len(df), index=df.index)
    r_break = pd.Series([False] * len(df), index=df.index)

    # Retest event tracking
    s_ret_valid = pd.Series([False] * len(df), index=df.index)
    r_ret_valid = pd.Series([False] * len(df), index=df.index)
    s_ret_occurred = False
    r_ret_occurred = False

    prev_pl = np.nan
    prev_ph = np.nan

    for i in range(bb, len(df)):
        curr_pl = pl.iloc[i]
        curr_ph = ph_vals.iloc[i]

        if not pd.isna(curr_pl) and not pd.isna(prev_pl) and curr_pl != prev_pl:
            s_break.iloc[i] = False
        if not pd.isna(curr_ph) and not pd.isna(prev_ph) and curr_ph != prev_ph:
            r_break.iloc[i] = False

        s_bot_val = s_bot.iloc[i] if not pd.isna(s_bot.iloc[i]) else np.nan
        s_top_val = s_top.iloc[i] if not pd.isna(s_top.iloc[i]) else np.nan
        r_bot_val = r_bot.iloc[i] if not pd.isna(r_bot.iloc[i]) else np.nan
        r_top_val = r_top.iloc[i] if not pd.isna(r_top.iloc[i]) else np.nan

        if not pd.isna(s_bot_val):
            if close.iloc[i] < s_bot_val and (i == 0 or close.iloc[i - 1] >= s_bot_val):
                s_break.iloc[i] = True

        if not pd.isna(r_top_val):
            if close.iloc[i] > r_top_val and (i == 0 or close.iloc[i - 1] <= r_top_val):
                r_break.iloc[i] = True

        if not pd.isna(curr_pl) and pd.isna(prev_pl):
            if not s_break.iloc[i]:
                s_break.iloc[i] = False
        if not pd.isna(curr_ph) and pd.isna(prev_ph):
            if not r_break.iloc[i]:
                r_break.iloc[i] = False

        prev_pl = curr_pl
        prev_ph = curr_ph

        if s_break.iloc[i]:
            s1 = high.iloc[i] >= s_top_val and close.iloc[i] <= s_bot_val
            s2 = high.iloc[i] >= s_top_val and close.iloc[i] >= s_bot_val and close.iloc[i] <= s_top_val
            s3 = high.iloc[i] >= s_bot_val and high.iloc[i] <= s_top_val
            s4 = high.iloc[i] >= s_bot_val and high.iloc[i] <= s_top_val and close.iloc[i] < s_bot_val
            s_ret_active = s1 or s2 or s3 or s4

            if s_ret_active and (i == 0 or not s_ret_active):
                bars_since_break = 0
            else:
                bars_since_break = 0
                for j in range(i - 1, -1, -1):
                    if s_break.iloc[j]:
                        break
                    bars_since_break += 1

            if bars_since_break > input_retSince:
                if s1:
                    ret_cond = close.iloc[i] <= s_bot.iloc[i] if not pd.isna(s_bot.iloc[i]) else False
                elif s2:
                    ret_cond = close.iloc[i] >= s_bot.iloc[i] and close.iloc[i] <= s_top.iloc[i] if not pd.isna(s_bot.iloc[i]) and not pd.isna(s_top.iloc[i]) else False
                elif s3:
                    ret_cond = True
                elif s4:
                    ret_cond = close.iloc[i] < s_bot.iloc[i] if not pd.isna(s_bot.iloc[i]) else False
                else:
                    ret_cond = False

                bars_since_ret = 0
                for j in range(i - 1, -1, -1):
                    if s_ret_active and (j == 0 or not (high.iloc[j] >= s_bot.iloc[j] if not pd.isna(s_bot.iloc[j]) else False)):
                        break
                    bars_since_ret += 1

                if bars_since_ret > 0 and bars_since_ret <= input_retValid and ret_cond and not s_ret_occurred:
                    s_ret_valid.iloc[i] = True
                    s_ret_occurred = True

        if r_break.iloc[i]:
            r1 = low.iloc[i] <= r_bot_val and close.iloc[i] >= r_top_val
            r2 = low.iloc[i] <= r_bot_val and close.iloc[i] <= r_top_val and close.iloc[i] >= r_bot_val
            r3 = low.iloc[i] <= r_top_val and low.iloc[i] >= r_bot_val
            r4 = low.iloc[i] <= r_top_val and low.iloc[i] >= r_bot_val and close.iloc[i] > r_top_val
            r_ret_active = r1 or r2 or r3 or r4

            if r_ret_active and (i == 0 or not r_ret_active):
                bars_since_break = 0
            else:
                bars_since_break = 0
                for j in range(i - 1, -1, -1):
                    if r_break.iloc[j]:
                        break
                    bars_since_break += 1

            if bars_since_break > input_retSince:
                if r1:
                    ret_cond = close.iloc[i] >= r_top.iloc[i] if not pd.isna(r_top.iloc[i]) else False
                elif r2:
                    ret_cond = close.iloc[i] <= r_top.iloc[i] and close.iloc[i] >= r_bot.iloc[i] if not pd.isna(r_top.iloc[i]) and not pd.isna(r_bot.iloc[i]) else False
                elif r3:
                    ret_cond = True
                elif r4:
                    ret_cond = close.iloc[i] > r_top.iloc[i] if not pd.isna(r_top.iloc[i]) else False
                else:
                    ret_cond = False

                bars_since_ret = 0
                for j in range(i - 1, -1, -1):
                    if r_ret_active and (j == 0 or not (low.iloc[j] <= r_bot.iloc[j] if not pd.isna(r_bot.iloc[j]) else False)):
                        break
                    bars_since_ret += 1

                if bars_since_ret > 0 and bars_since_ret <= input_retValid and ret_cond and not r_ret_occurred:
                    r_ret_valid.iloc[i] = True
                    r_ret_occurred = True

        if s_break.iloc[i]:
            s_ret_occurred = False
        if r_break.iloc[i]:
            r_ret_occurred = False

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if s_ret_valid.iloc[i] and not pd.isna(s_bot.iloc[i]):
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

        if r_ret_valid.iloc[i] and not pd.isna(r_top.iloc[i]):
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

    return entries