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
    open_price = df['open']
    time_col = df['time']

    atr_period = 200
    atr_multi = 0.25

    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    if len(tr) > 0:
        tr.iloc[0] = high.iloc[0] - low.iloc[0]
    atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()
    atr_value = atr * atr_multi
    expanding_mean = (high - low).expanding().mean()
    atr_value = atr_value.fillna(expanding_mean)

    fvg_up = (low > high.shift(2)) & (close.shift(1) > high.shift(2))
    fvg_down = (high < low.shift(2)) & (close.shift(1) < low.shift(2))

    bull_fvg_ary = []
    bear_fvg_ary = []
    bull_inv_ary = []
    bear_inv_ary = []

    buffer_size = 100
    trade_num = 1
    entries = []

    signal_pref = "Close"
    wt = signal_pref == "Wick"

    for i in range(2, len(df)):
        if fvg_up.iloc[i] and abs(low.iloc[i] - high.shift(2).iloc[i]) > atr_value.iloc[i]:
            if len(bull_fvg_ary) >= buffer_size:
                bull_fvg_ary.pop(0)
            bull_fvg_ary.append({
                'left': time_col.shift(1).iloc[i],
                'top': high.shift(2).iloc[i],
                'right': time_col.iloc[i],
                'bot': low.iloc[i],
                'mid': (low.iloc[i] + high.shift(2).iloc[i]) / 2.0,
                'dir': 1,
                'state': 0,
                'labs': [],
                'x_val': None
            })

        if fvg_down.iloc[i] and abs(low.shift(2).iloc[i] - high.iloc[i]) > atr_value.iloc[i]:
            if len(bear_fvg_ary) >= buffer_size:
                bear_fvg_ary.pop(0)
            bear_fvg_ary.append({
                'left': time_col.shift(1).iloc[i],
                'top': high.iloc[i],
                'right': time_col.iloc[i],
                'bot': low.shift(2).iloc[i],
                'mid': (high.iloc[i] + low.shift(2).iloc[i]) / 2.0,
                'dir': -1,
                'state': 0,
                'labs': [],
                'x_val': None
            })

        for j in range(len(bull_fvg_ary) - 1, -1, -1):
            fvg = bull_fvg_ary[j]
            c_bot = min(open_price.iloc[i], close.iloc[i])
            if fvg['dir'] == 1 and c_bot < fvg['bot']:
                fvg['x_val'] = time_col.iloc[i]
                bull_inv_ary.append(fvg)
                bull_fvg_ary.pop(j)

        for j in range(len(bear_fvg_ary) - 1, -1, -1):
            fvg = bear_fvg_ary[j]
            c_top = max(open_price.iloc[i], close.iloc[i])
            if fvg['dir'] == -1 and c_top > fvg['top']:
                fvg['x_val'] = time_col.iloc[i]
                bear_inv_ary.append(fvg)
                bear_fvg_ary.pop(j)

        for j in range(len(bull_inv_ary) - 1, -1, -1):
            fvg = bull_inv_ary[j]
            bx_top = fvg['top']
            bx_bot = fvg['bot']
            _dir = fvg['dir']
            st = fvg['state']

            if st == 0 and _dir == 1:
                fvg['state'] = 1
                fvg['dir'] = -1
            if _dir == -1 and st == 0:
                fvg['state'] = 1
                fvg['dir'] = 1
            if st >= 1:
                fvg['right'] = time_col.iloc[i]

            test_val_bear = high.iloc[i] if wt else close.shift(1).iloc[i]
            if _dir == -1 and st == 1 and close.iloc[i] < bx_bot and test_val_bear >= bx_bot and test_val_bear < bx_top:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': time_col.iloc[i],
                    'entry_time': datetime.fromtimestamp(time_col.iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close.iloc[i],
                    'raw_price_b': close.iloc[i]
                })
                trade_num += 1
                bull_inv_ary.pop(j)
                continue

            test_val_bull = low.iloc[i] if wt else close.shift(1).iloc[i]
            if _dir == 1 and st == 1 and close.iloc[i] > bx_top and test_val_bull <= bx_top and test_val_bull > bx_bot:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': time_col.iloc[i],
                    'entry_time': datetime.fromtimestamp(time_col.iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close.iloc[i],
                    'raw_price_b': close.iloc[i]
                })
                trade_num += 1
                bull_inv_ary.pop(j)
                continue

            c_top = max(open_price.iloc[i], close.iloc[i])
            c_bot = min(open_price.iloc[i], close.iloc[i])
            if st >= 1 and ((_dir == -1 and c_top > bx_top) or (_dir == 1 and c_bot < bx_bot)):
                bull_inv_ary.pop(j)
                continue

        for j in range(len(bear_inv_ary) - 1, -1, -1):
            fvg = bear_inv_ary[j]
            bx_top = fvg['top']
            bx_bot = fvg['bot']
            _dir = fvg['dir']
            st = fvg['state']

            if st == 0 and _dir == 1:
                fvg['state'] = 1
                fvg['dir'] = -1