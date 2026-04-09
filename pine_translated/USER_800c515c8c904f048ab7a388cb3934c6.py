import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    BarBackCheck = 5
    CISDVal = 25
    SwingPeriod = 50
    MaxSwingBack = 100
    BackToBreakPeriod = CISDVal

    def wilder_atr(data, period):
        high = data['high']
        low = data['low']
        close = data['close']
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr

    def pivot_high(prices, left_bars, right_bars):
        pivots = np.full(len(prices), np.nan)
        for i in range(left_bars, len(prices) - right_bars):
            window_left = prices.iloc[i - left_bars:i]
            window_right = prices.iloc[i + 1:i + right_bars + 1]
            if prices.iloc[i] >= window_left.max() and prices.iloc[i] >= window_right.max():
                pivots[i] = prices.iloc[i]
        return pd.Series(pivots, index=prices.index)

    def pivot_low(prices, left_bars, right_bars):
        pivots = np.full(len(prices), np.nan)
        for i in range(left_bars, len(prices) - right_bars):
            window_left = prices.iloc[i - left_bars:i]
            window_right = prices.iloc[i + 1:i + right_bars + 1]
            if prices.iloc[i] <= window_left.min() and prices.iloc[i] <= window_right.min():
                pivots[i] = prices.iloc[i]
        return pd.Series(pivots, index=prices.index)

    atr = wilder_atr(df, 55)
    MSH = pivot_high(df['high'], SwingPeriod, SwingPeriod)
    MSL = pivot_low(df['low'], SwingPeriod, SwingPeriod)

    Z = 10**17
    ms_h_p = []
    ms_h_i = []
    ms_l_p = []
    ms_l_i = []
    aoi_msh = []
    aoi_msl = []
    permit_msh = []
    permit_msl = []

    cisd_lvl_h = 0.0
    cisd_bar_h = 0
    cisd_lvl_l = 0.0
    cisd_bar_l = 0
    permit_h_set = True
    permit_l_set = True
    bull_cisd_active = False
    bear_cisd_active = False

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if not pd.isna(MSH.iloc[i]):
            ms_h_p.append(df['high'].iloc[i - SwingPeriod] if i >= SwingPeriod else np.nan)
            ms_h_i.append(df['time'].iloc[i - SwingPeriod] if i >= SwingPeriod else 0)
            permit_msh.append(True)
            aoi_msh.append(Z)

        if not pd.isna(MSL.iloc[i]):
            ms_l_p.append(df['low'].iloc[i - SwingPeriod] if i >= SwingPeriod else np.nan)
            ms_l_i.append(df['time'].iloc[i - SwingPeriod] if i >= SwingPeriod else 0)
            permit_msl.append(True)
            aoi_msl.append(Z)

        h_alert = False
        l_alert = False
        hswingback = min(len(ms_h_p), MaxSwingBack)
        lswingback = min(len(ms_l_p), MaxSwingBack)

        for j in range(1, hswingback + 1):
            idx = len(ms_h_p) - j
            if idx >= 0 and permit_msh[idx] and len(aoi_msh) > idx and aoi_msh[idx] == Z:
                msh_val = ms_h_p[idx]
                if not pd.isna(msh_val):
                    real_liq = msh_val < df['high'].iloc[i] and msh_val > df['close'].iloc[i]
                    considerable_liq = (msh_val > df['close'].iloc[i] and msh_val < df['close'].iloc[i-1]) and df['close'].iloc[i] < df['open'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i-1]
                    if real_liq or considerable_liq:
                        permit_msh[idx] = False
                        h_alert = True
                    if msh_val < df['close'].iloc[i] and aoi_msh[idx] == Z:
                        aoi_msh[idx] = i

        for j in range(1, lswingback + 1):
            idx = len(ms_l_p) - j
            if idx >= 0 and permit_msl[idx] and len(aoi_msl) > idx and aoi_msl[idx] == Z:
                msl_val = ms_l_p[idx]
                if not pd.isna(msl_val):
                    real_liq_l = msl_val > df['low'].iloc[i] and msl_val < df['close'].iloc[i]
                    considerable_liq_l = (msl_val < df['close'].iloc[i] and msl_val > df['close'].iloc[i-1]) and df['close'].iloc[i] > df['open'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i-1]
                    if real_liq_l or considerable_liq_l:
                        permit_msl[idx] = False
                        l_alert = True
                    if msl_val > df['close'].iloc[i] and aoi_msl[idx] == Z:
                        aoi_msl[idx] = i

        body = df['close'].iloc[i] - df['open'].iloc[i]
        if permit_h_set:
            bear_high = max(df['high'].iloc[i], df['high'].iloc[i-1] if i > 0 else df['high'].iloc[i], df['high'].iloc[i-2] if i > 1 else df['high'].iloc[i])
            bear_low = df['low'].iloc[i]
            bear_bar = i
            for j in range(1, BarBackCheck + 1):
                if i - j >= 0 and body < 0:
                    if BarBackCheck > 1 and j > 1:
                        cisd_lvl_h = min(df['open'].iloc[i-j], df['open'].iloc[i-j+1] if i-j+1 <= i else df['open'].iloc[i-j])
                        cisd_bar_h = i - j + 1 if df['open'].iloc[i-j] != cisd_lvl_h else i - j
                    else:
                        cisd_lvl_h = df['open'].iloc[i-j]
                        cisd_bar_h = i - j
                    permit_h_set = False
                    bull_cisd_active = True
                    break

        if permit_l_set:
            bull_high = df['high'].iloc[i]
            bull_low = min(df['low'].iloc[i], df['low'].iloc[i-1] if i > 0 else df['low'].iloc[i], df['low'].iloc[i-2] if i > 1 else df['low'].iloc[i])
            bull_bar = i
            for j in range(1, BarBackCheck + 1):
                if i - j >= 0 and body > 0:
                    if BarBackCheck > 1 and j > 1:
                        cisd_lvl_l = max(df['open'].iloc[i-j], df['open'].iloc[i-j+1] if i-j+1 <= i else df['open'].iloc[i-j])
                        cisd_bar_l = i - j + 1 if df['open'].iloc[i-j] != cisd_lvl_l else i - j
                    else:
                        cisd_lvl_l = df['open'].iloc[i-j]
                        cisd_bar_l = i - j
                    permit_l_set = False
                    bear_cisd_active = True
                    break

        if bull_cisd_active and df['close'].iloc[i] >= cisd_lvl_h and (i - cisd_bar_h) <= CISDVal:
            if l_alert:
                ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': entry_time,
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                })
                trade_num += 1

        if bear_cisd_active and df['close'].iloc[i] <= cisd_lvl_l and (i - cisd_bar_l) <= CISDVal:
            if h_alert:
                ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': entry_time,
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                })
                trade_num += 1

    return entries