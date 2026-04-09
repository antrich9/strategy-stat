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
    # Parameters (matching default inputs from Pine Script)
    PeriodE2PSS = 15
    useE2PSS = True
    inverseE2PSS = False

    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0

    HEV_length = 200
    HV_ma = 20
    divisor = 3.6

    useHEV = True
    crossHEV = True
    inverseHEV = False
    highlightMovementsHEV = True

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    hl2 = (df['high'] + df['low']) / 2.0

    # --- E2PSS Filter ---
    pi = 2 * np.pi
    a1 = np.exp(-1.414 * pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3

    Filt2 = pd.Series(np.nan, index=df.index)
    Filt2.iloc[0] = hl2.iloc[0]
    if len(df) > 1:
        Filt2.iloc[1] = hl2.iloc[1]
    if len(df) > 2:
        Filt2.iloc[2] = hl2.iloc[2]

    PriceE2PSS = hl2
    for i in range(3, len(df)):
        prev1 = Filt2.iloc[i-1] if not np.isnan(Filt2.iloc[i-1]) else PriceE2PSS.iloc[i-1]
        prev2 = Filt2.iloc[i-2] if not np.isnan(Filt2.iloc[i-2]) else PriceE2PSS.iloc[i-2]
        Filt2.iloc[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * prev1 + coef3 * prev2

    TriggerE2PSS = Filt2.shift(1).fillna(0)

    signalLongE2PSS = Filt2 > TriggerE2PSS
    signalShortE2PSS = Filt2 < TriggerE2PSS

    if inverseE2PSS:
        signalLongE2PSSFinal = signalShortE2PSS
        signalShortE2PSSFinal = signalLongE2PSS
    else:
        signalLongE2PSSFinal = signalLongE2PSS
        signalShortE2PSSFinal = signalShortE2PSS

    # --- Trendilo ---
    pct_change = close.diff(trendilo_smooth) / close * 100

    def alma(src, length, offset, sigma):
        m = offset * (length - 1)
        s = sigma * (length - 1) / 6.0
        weights = np.exp(-np.square(np.arange(length) - m) / (2 * s * s))
        weights = weights / weights.sum()
        return pd.Series(np.convolve(src, weights, mode='valid')).reindex(src.index, method='ffill')

    avg_pct_change = alma(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    rms_vals = []
    for i in range(len(df)):
        if i < trendilo_length - 1 or np.isnan(avg_pct_change.iloc[i]):
            rms_vals.append(np.nan)
        else:
            window = avg_pct_change.iloc[i-trendilo_length+1:i+1]
            if window.isna().any():
                rms_vals.append(np.nan)
            else:
                rms_vals.append(trendilo_bmult * np.sqrt((window**2).sum() / trendilo_length))
    rms = pd.Series(rms_vals, index=df.index)

    trendilo_dir = pd.Series(0, index=df.index)
    trendilo_dir[avg_pct_change > rms] = 1
    trendilo_dir[avg_pct_change < -rms] = -1

    # --- HawkEye Volume ---
    range_1 = high - low
    rangeAvg = range_1.rolling(HEV_length).mean()
    durchschnitt = volume.rolling(HV_ma).mean()
    volumeA = volume.rolling(HEV_length).mean()

    high1 = high.shift(1)
    low1 = low.shift(1)
    mid1 = hl2.shift(1)

    u1 = mid1 + (high1 - low1) / divisor
    d1 = mid1 - (high1 - low1) / divisor

    r_enabled1 = (range_1 > rangeAvg) & (close < d1) & (volume > volumeA)
    r_enabled2 = close < mid1
    r_enabled = r_enabled1 | r_enabled2

    g_enabled1 = close > mid1
    g_enabled2 = (range_1 > rangeAvg) & (close > u1) & (volume > volumeA)
    g_enabled3 = (high > high1) & (range_1 < rangeAvg / 1.5) & (volume < volumeA)
    g_enabled4 = (low < low1) & (range_1 < rangeAvg / 1.5) & (volume > volumeA)
    g_enabled = g_enabled1 | g_enabled2 | g_enabled3 | g_enabled4

    gr_enabled1 = (range_1 > rangeAvg) & (close > d1) & (close < u1) & (volume > volumeA) & (volume < volumeA * 1.5) & (volume > volume.shift(1))
    gr_enabled2 = (range_1 < rangeAvg / 1.5) & (volume < volumeA / 1.5)
    gr_enabled3 = (close > d1) & (close < u1)
    gr_enabled = gr_enabled1 | gr_enabled2 | gr_enabled3

    basicLongHEVCondition = g_enabled & (volume > durchschnitt)
    basicShorHEVondition = r_enabled & (volume > durchschnitt)

    if useHEV:
        if highlightMovementsHEV:
            HEVSignalsLong = basicLongHEVCondition
            HEVSignalsShort = basicShorHEVondition
        else:
            HEVSignalsLong = g_enabled
            HEVSignalsShort = r_enabled
    else:
        HEVSignalsLong = pd.Series(True, index=df.index)
        HEVSignalsShort = pd.Series(True, index=df.index)

    HEVSignalsLongPrev = HEVSignalsLong.shift(1).fillna(False)
    HEVSignalsShortPrev = HEVSignalsShort.shift(1).fillna(False)

    if crossHEV:
        HEVSignalsLongCross = (~HEVSignalsLongPrev) & HEVSignalsLong
        HEVSignalsShorHEVross = (~HEVSignalsShortPrev) & HEVSignalsShort
    else:
        HEVSignalsLongCross = HEVSignalsLong
        HEVSignalsShorHEVross = HEVSignalsShort

    if inverseHEV:
        HEVSignalsLongFinal = HEVSignalsShorHEVross
        HEVSignalsShortFinal = HEVSignalsLongCross
    else:
        HEVSignalsLongFinal = HEVSignalsLongCross
        HEVSignalsShortFinal = HEVSignalsShorHEVross

    # --- Entry Conditions ---
    long_condition = signalLongE2PSSFinal & (trendilo_dir == 1) & HEVSignalsLongFinal
    short_condition = signalShortE2PSSFinal & (trendilo_dir == -1) & HEVSignalsShortFinal

    # --- Generate Entries ---
    entries = []
    trade_num = 1
    in_position = False

    for i in range(1, len(df)):
        if in_position:
            continue

        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
            in_position = True
        elif short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
            in_position = True

    return entries