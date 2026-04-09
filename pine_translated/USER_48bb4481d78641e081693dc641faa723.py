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
    if len(df) < 25:
        return []

    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    close_arr = df['close'].values
    time_arr = df['time'].values

    tf_o = []
    tf_h = []
    tf_l = []
    tf_c = []
    tf_time = []
    tf_t = []

    entries = []
    trade_num = 1

    disp_x = 3

    def avg_over(h, l, length):
        return np.mean(h[:length] - l[:length])

    for i in range(len(df)):
        tf_o.insert(0, open_arr[i])
        tf_h.insert(0, high_arr[i])
        tf_l.insert(0, low_arr[i])
        tf_c.insert(0, close_arr[i])
        tf_time.insert(0, time_arr[i])
        tf_t.insert(0, True)

        if len(tf_o) > 300:
            tf_o.pop()
            tf_h.pop()
            tf_l.pop()
            tf_c.pop()
            tf_t.pop()
            tf_time.pop()

        if len(tf_o) > 20:
            bull = tf_c[1] > tf_o[1]
            if bull:
                fvg = (tf_h[2] < tf_l[0] and tf_l[1] <= tf_h[2] and tf_h[1] >= tf_l[0])
                fvg_len = tf_l[0] - tf_h[2]
            else:
                fvg = (tf_l[2] > tf_h[0] and tf_l[1] <= tf_h[0] and tf_l[2] <= tf_h[1])
                fvg_len = tf_l[2] - tf_h[0]

            atr_check = fvg_len > avg_over(tf_h, tf_l, 20) * disp_x / 10

            if tf_t[2] and fvg and atr_check:
                if bull:
                    top = tf_l[0]
                    bottom = tf_h[2]
                else:
                    top = tf_l[2]
                    bottom = tf_h[0]

                if len(tf_o) > 1:
                    tf_o.pop(1)
                    tf_h.pop(1)
                    tf_l.pop(1)
                    tf_c.pop(1)
                    tf_t.pop(1)
                    tf_time.pop(1)

            reg_fvg_top = []
            reg_fvg_bottom = []
            reg_fvg_side = []
            inv_fvg_top = []
            inv_fvg_bottom = []
            inv_fvg_side = []

            for j in range(len(reg_fvg_side) - 1, -1, -1):
                remove_bull = reg_fvg_side[j] == 'bull' and tf_c[0] < reg_fvg_bottom[j]
                remove_bear = reg_fvg_side[j] == 'bear' and tf_c[0] > reg_fvg_top[j]

                if remove_bull or remove_bear:
                    inv_fvg_top.append(reg_fvg_top[j])
                    inv_fvg_bottom.append(reg_fvg_bottom[j])
                    if remove_bear:
                        inv_fvg_side.append('inv bear')
                    else:
                        inv_fvg_side.append('inv bull')

                    reg_fvg_top.pop(j)
                    reg_fvg_bottom.pop(j)
                    reg_fvg_side.pop(j)

            disp_limit = 10
            while len(reg_fvg_side) > disp_limit:
                reg_fvg_top.pop()
                reg_fvg_bottom.pop()
                reg_fvg_side.pop()

            for j in range(len(inv_fvg_side) - 1, -1, -1):
                if inv_fvg_side[j] == 'inv bull' and tf_c[0] > inv_fvg_top[j]:
                    entry_ts = tf_time[0]
                    entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                    entry_price = tf_c[0]
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
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
                    inv_fvg_top.pop(j)
                    inv_fvg_bottom.pop(j)
                    inv_fvg_side.pop(j)

                if inv_fvg_side[j] == 'inv bear' and tf_c[0] < inv_fvg_bottom[j]:
                    entry_ts = tf_time[0]
                    entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                    entry_price = tf_c[0]
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
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
                    inv_fvg_top.pop(j)
                    inv_fvg_bottom.pop(j)
                    inv_fvg_side.pop(j)

    return entries