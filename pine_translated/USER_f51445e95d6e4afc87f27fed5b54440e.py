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
    # Input parameters
    useE2PSS = True
    inverseE2PSS = False
    PeriodE2PSS = 15
    
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    useHEV = True
    crossHEV = True
    inverseHEV = False
    highlightMovementsHEV = True
    
    length = 200
    HV_ma = 20
    divisor = 3.6
    
    # Initialize result list
    entries = []
    trade_num = 1
    
    # Ensure data has enough rows
    if len(df) < 3:
        return entries
    
    # Calculate E2PSS
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * np.pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    # Calculate Filt2 using recursive formula
    Filt2 = np.zeros(len(df))
    Filt2[0] = df['close'].iloc[0]
    Filt2[1] = df['close'].iloc[1]
    for i in range(2, len(df)):
        Filt2[i] = coef1 * df['hl2'].iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
    
    TriggerE2PSS = pd.Series(Filt2).shift(1).fillna(0).values
    
    # Signal conditions for E2PSS
    if useE2PSS:
        signalLongE2PSS = Filt2 > TriggerE2PSS
        signalShortE2PSS = Filt2 < TriggerE2PSS
    else:
        signalLongE2PSS = np.ones(len(df), dtype=bool)
        signalShortE2PSS = np.ones(len(df), dtype=bool)
    
    if inverseE2PSS:
        signalLongE2PSSFinal = signalShortE2PSS.copy()
        signalShortE2PSSFinal = signalLongE2PSS.copy()
    else:
        signalLongE2PSSFinal = signalLongE2PSS.copy()
        signalShortE2PSSFinal = signalShortE2PSS.copy()
    
    # Calculate Trendilo
    pct_change = df['close'].diff(trendilo_smooth) / df['close'] * 100
    
    # ALMA approximation using numpy convolution
    def alma_approx(series, length, offset, sigma):
        m = np.floor(offset * (length - 1))
        s = length / sigma if sigma > 0 else length
        w = np.exp(-np.arange(length)**2 / (2 * s * s))
        w = w / w.sum()
        result = np.convolve(series, w, mode='valid')
        pad = np.empty(length - 1)
        pad[:] = np.nan
        return np.concatenate([pad, result])
    
    avg_pct_change = alma_approx(pct_change.values, trendilo_length, trendilo_offset, trendilo_sigma)
    
    # Calculate RMS
    rms = np.zeros(len(df))
    for i in range(trendilo_length - 1, len(df)):
        window = avg_pct_change[i - trendilo_length + 1:i + 1]
        rms[i] = trendilo_bmult * np.sqrt(np.mean(window * window))
    
    # Trendilo direction
    trendilo_dir = np.zeros(len(df))
    trendilo_dir[avg_pct_change > rms] = 1
    trendilo_dir[avg_pct_change < -rms] = -1
    
    # Calculate HEV components
    range_1 = df['high'] - df['low']
    rangeAvg = pd.Series(range_1).rolling(length).mean()
    durchschnitt = pd.Series(df['volume']).rolling(HV_ma).mean()
    volumeA = pd.Series(df['volume']).rolling(length).mean()
    
    high1 = df['high'].shift(1)
    low1 = df['low'].shift(1)
    mid1 = df['hl2'].shift(1)
    
    u1 = mid1 + (high1 - low1) / divisor
    d1 = mid1 - (high1 - low1) / divisor
    
    # Green conditions (g_enabled)
    g_enabled1 = df['close'] > mid1
    g_enabled2 = (range_1 > rangeAvg) & (df['close'] > u1) & (df['volume'] > volumeA)
    g_enabled3 = (df['high'] > high1) & (range_1 < rangeAvg / 1.5) & (df['volume'] < volumeA)
    g_enabled4 = (df['low'] < low1) & (range_1 < rangeAvg / 1.5) & (df['volume'] > volumeA)
    g_enabled = g_enabled1 | g_enabled2 | g_enabled3 | g_enabled4
    
    # Red conditions (r_enabled)
    r_enabled1 = (range_1 > rangeAvg) & (df['close'] < d1) & (df['volume'] > volumeA)
    r_enabled2 = df['close'] < mid1
    r_enabled = r_enabled1 | r_enabled2
    
    # Gray conditions (gr_enabled)
    gr_enabled1 = (range_1 > rangeAvg) & (df['close'] > d1) & (df['close'] < u1) & (df['volume'] > volumeA) & (df['volume'] < volumeA * 1.5) & (df['volume'] > df['volume'].shift(1))
    gr_enabled2 = (range_1 < rangeAvg / 1.5) & (df['volume'] < volumeA / 1.5)
    gr_enabled3 = (df['close'] > d1) & (df['close'] < u1)
    gr_enabled = gr_enabled1 | gr_enabled2 | gr_enabled3
    
    # Basic HEV conditions
    basicLongHEVCondition = g_enabled & (df['volume'] > durchschnitt)
    basicShorHEVondition = r_enabled & (df['volume'] > durchschnitt)
    
    # HEV signals
    if useHEV:
        HEVSignalsLong = basicLongHEVCondition if highlightMovementsHEV else g_enabled
        HEVSignalsShort = basicShorHEVondition if highlightMovementsHEV else r_enabled
    else:
        HEVSignalsLong = pd.Series(np.ones(len(df), dtype=bool))
        HEVSignalsShort = pd.Series(np.ones(len(df), dtype=bool))
    
    # Cross confirmation
    HEVSignalsLong_prev = HEVSignalsLong.shift(1).fillna(False)
    HEVSignalsShort_prev = HEVSignalsShort.shift(1).fillna(False)
    
    if crossHEV:
        HEVSignalsLongCross = (~HEVSignalsLong_prev) & HEVSignalsLong
        HEVSignalsShorHEVross = (~HEVSignalsShort_prev) & HEVSignalsShort
    else:
        HEVSignalsLongCross = HEVSignalsLong
        HEVSignalsShorHEVross = HEVSignalsShort
    
    # Final HEV signals
    if inverseHEV:
        HEVSignalsLongFinal = HEVSignalsShorHEVross
        HEVSignalsShortFinal = HEVSignalsLongCross
    else:
        HEVSignalsLongFinal = HEVSignalsLongCross
        HEVSignalsShortFinal = HEVSignalsShorHEVross
    
    # Long condition
    long_condition = signalLongE2PSSFinal & (trendilo_dir == 1) & basicLongHEVCondition
    
    # Short condition
    short_condition = signalShortE2PSSFinal & (trendilo_dir == -1) & basicShorHEVondition
    
    # Generate entries
    for i in range(1, len(df)):
        if pd.isna(Filt2[i]) or pd.isna(avg_pct_change[i]) or pd.isna(rangeAvg.iloc[i]) or pd.isna(volumeA.iloc[i]):
            continue
        
        entry_price = df['close'].iloc[i]
        ts = df['time'].iloc[i]
        
        if long_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
        
        if short_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
    
    return entries