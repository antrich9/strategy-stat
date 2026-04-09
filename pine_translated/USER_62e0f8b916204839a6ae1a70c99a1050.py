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
    # Parameters
    atrLength = 14
    atrMultiplier = 1.5
    lengthCmo = 14
    alpha = 0.2
    cmoLength = 14
    volatilityRatioLength = 20
    volatilityThreshold = 1.5
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    input_repType = 'On'
    startHour = 7
    endHour = 18

    def Wilder_sum(series, period):
        alpha = 1.0 / period
        return series.ewm(alpha=alpha, adjust=False).mean()

    def calc_wilder_atr(df, period):
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        return atr

    def calc_cmo(src, period):
        diff = src.diff()
        up = diff.clip(lower=0)
        down = (-diff).clip(lower=0)
        up_sum = Wilder_sum(up, period)
        down_sum = Wilder_sum(down, period)
        cmo = 100 * (up_sum - down_sum) / (up_sum + down_sum)
        return cmo

    def calc_vidya(src, period_cmo, alpha):
        cmo_val = calc_cmo(src, period_cmo)
        vidya = pd.Series(index=src.index, dtype=float)
        vidya.iloc[0] = src.iloc[0]
        for i in range(1, len(src)):
            vidya.iloc[i] = vidya.iloc[i-1] + alpha * (cmo_val.iloc[i] / 100) * (src.iloc[i] - vidya.iloc[i-1])
        return vidya

    def calc_sma(src, period):
        return src.rolling(window=period).mean()

    bb = input_lookback

    vidya = calc_vidya(df['close'], lengthCmo, alpha)
    averageVolume = calc_sma(df['volume'], volatilityRatioLength)
    volumeVolatilityRatio = df['volume'] / averageVolume
    atr = calc_wilder_atr(df, atrLength)

    pl = pd.Series(index=df.index, dtype=float)
    ph = pd.Series(index=df.index, dtype=float)
    for i in range(bb, len(df) - bb):
        pl.iloc[i] = df['low'].iloc[i - bb]
        ph.iloc[i] = df['high'].iloc[i - bb]
        for j in range(1, bb + 1):
            if df['low'].iloc[i - bb + j] < pl.iloc[i]:
                pl.iloc[i] = df['low'].iloc[i - bb + j]
            if df['high'].iloc[i - bb + j] > ph.iloc[i]:
                ph.iloc[i] = df['high'].iloc[i - bb + j]

    pl_filled = pl.ffill()
    ph_filled = ph.ffill()

    s_yLoc = pd.Series(index=df.index, dtype=float)
    r_yLoc = pd.Series(index=df.index, dtype=float)
    for i in range(bb + 1, len(df)):
        s_yLoc.iloc[i] = df['low'].iloc[bb - 1] if df['low'].iloc[bb - 1] < df['low'].iloc[bb + 1] else df['low'].iloc[bb + 1]
        r_yLoc.iloc[i] = df['high'].iloc[bb + 1] if df['high'].iloc[bb + 1] > df['high'].iloc[bb - 1] else df['high'].iloc[bb - 1]

    sTop = s_yLoc
    rTop = ph_filled
    sBot = pl_filled
    rBot = r_yLoc

    def repaint(c1, c2, c3):
        if input_repType == 'On':
            return c1
        elif input_repType == 'Off: High & Low':
            return c2
        else:
            return c3

    cu_raw = df['close'] < sBot
    cu_raw_shifted = df['low'] < sBot
    cu_confirmed = (df['close'] < sBot) & (df['close'].shift(1) >= sBot.shift(1))
    cu = repaint(cu_raw, cu_raw_shifted, cu_confirmed)

    co_raw = df['close'] > rTop
    co_raw_shifted = df['high'] > rTop
    co_confirmed = (df['close'] > rTop) & (df['close'].shift(1) <= rTop.shift(1))
    co = repaint(co_raw, co_raw_shifted, co_confirmed)

    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)

    for i in range(bb + 1, len(df)):
        if cu.iloc[i] and pd.isna(sBreak.iloc[i - 1] if i > 0 else False):
            sBreak.iloc[i] = True
        if co.iloc[i] and pd.isna(rBreak.iloc[i - 1] if i > 0 else False):
            rBreak.iloc[i] = True
        if i > 0:
            if pl.iloc[i] != pl.iloc[i - 1]:
                sBreak.iloc[i] = np.nan
            if ph.iloc[i] != ph.iloc[i - 1]:
                rBreak.iloc[i] = np.nan

    hour = pd.to_datetime(df['time'], unit='s').dt.hour
    in_trading_hours = (hour >= startHour) & (hour < endHour)

    bars_since_sbreak = pd.Series(np.nan, index=df.index)
    bars_since_rbreak = pd.Series(np.nan, index=df.index)
    for i in range(len(df)):
        for j in range(1, input_retSince + 10):
            if i - j >= 0:
                if sBreak.iloc[i - j]:
                    bars_since_sbreak.iloc[i] = j
                    break
        for j in range(1, input_retSince + 10):
            if i - j >= 0:
                if rBreak.iloc[i - j]:
                    bars_since_rbreak.iloc[i] = j
                    break

    s_ret1 = (bars_since_sbreak > input_retSince) & (df['high'] >= sTop) & (df['close'] <= sBot)
    s_ret2 = (bars_since_sbreak > input_retSince) & (df['high'] >= sTop) & (df['close'] >= sBot) & (df['close'] <= sTop)
    s_ret3 = (bars_since_sbreak > input_retSince) & (df['high'] >= sBot) & (df['high'] <= sTop)
    s_ret4 = (bars_since_sbreak > input_retSince) & (df['high'] >= sBot) & (df['high'] <= sTop) & (df['close'] < sBot)

    r_ret1 = (bars_since_rbreak > input_retSince) & (df['low'] <= rBot) & (df['close'] >= rTop)
    r_ret2 = (bars_since_rbreak > input_retSince) & (df['low'] <= rBot) & (df['close'] <= rTop) & (df['close'] >= rBot)
    r_ret3 = (bars_since_rbreak > input_retSince) & (df['low'] <= rTop) & (df['low'] >= rBot)
    r_ret4 = (bars_since_rbreak > input_retSince) & (df['low'] <= rTop) & (df['low'] >= rBot) & (df['close'] > rTop)

    vol_spike = volumeVolatilityRatio > volatilityThreshold

    s_ret_active = s_ret1 | s_ret2 | s_ret3 | s_ret4
    r_ret_active = r_ret1 | r_ret2 | r_ret3 | r_ret4

    s_ret_prev = s_ret_active.shift(1).fillna(False)
    r_ret_prev = r_ret_active.shift(1).fillna(False)

    s_ret_event = s_ret_active & ~s_ret_prev
    r_ret_event = r_ret_active & ~r_ret_prev

    retValue_s = pd.Series(np.nan, index=df.index)
    retValue_r = pd.Series(np.nan, index=df.index)
    retOccurred_s = pd.Series(False, index=df.index)
    retOccurred_r = pd.Series(False, index=df.index)

    for i in range(len(df)):
        if s_ret_event.iloc[i]:
            retValue_s.iloc[i] = sTop.iloc[i]
            retOccurred_s.iloc[i] = False
        else:
            if i > 0:
                retValue_s.iloc[i] = retValue_s.iloc[i - 1]
                retOccurred_s.iloc[i] = retOccurred_s.iloc[i - 1]
        if r_ret_event.iloc[i]:
            retValue_r.iloc[i] = rBot.iloc[i]
            retOccurred_r.iloc[i] = False
        else:
            if i > 0:
                retValue_r.iloc[i] = retValue_r.iloc[i - 1]
                retOccurred_r.iloc[i] = retOccurred_r.iloc[i - 1]

    retConditions_s = repaint(df['close'] <= retValue_s, df['high'] <= retValue_s, (df['close'] <= retValue_s))
    retConditions_r = repaint(df['close'] >= retValue_r, df['low'] >= retValue_r, (df['close'] >= retValue_r))

    retValid_s = (bars_since_sbreak > 0) & (bars_since_sbreak <= input_retValid) & retConditions_s & ~retOccurred_s & vol_spike
    retValid_r = (bars_since_rbreak > 0) & (bars_since_rbreak <= input_retValid) & retConditions_r & ~retOccurred_r & vol_spike

    long_entry = retValid_s & in_trading_hours
    short_entry = retValid_r & in_trading_hours

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if long_entry.iloc[i] and i >= bb:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        if short_entry.iloc[i] and i >= bb:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1

    return entries