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
    # Input parameters matching Pine Script defaults
    usejmaJMA = True
    usecolorjmaJMA = True
    inverseJMA = True
    lengthjmaJMA = 7
    phasejmaJMA = 50
    powerjmaJMA = 2
    
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    useStiffness = False
    maLengthStiffness = 100
    stiffLength = 60
    stiffSmooth = 3
    thresholdStiffness = 90
    
    close = df['close'].values
    n = len(close)
    
    # JMA calculations
    phasejmaJMARatiojmaJMA = 0.5 if phasejmaJMA < -100 else (2.5 if phasejmaJMA > 100 else phasejmaJMA / 100 + 1.5)
    betajmaJMA = 0.45 * (lengthjmaJMA - 1) / (0.45 * (lengthjmaJMA - 1) + 2)
    alphajmaJMA = betajmaJMA ** powerjmaJMA
    
    jmaJMA = np.zeros(n)
    e0JMA = 0.0
    e1JMA = 0.0
    e2JMA = 0.0
    prev_jmaJMA = 0.0
    
    for i in range(n):
        src = close[i]
        e0JMA = (1 - alphajmaJMA) * src + alphajmaJMA * e0JMA
        e1JMA = (src - e0JMA) * (1 - betajmaJMA) + betajmaJMA * e1JMA
        e2JMA = (e0JMA + phasejmaJMARatiojmaJMA * e1JMA - prev_jmaJMA) * (1 - alphajmaJMA) ** 2 + alphajmaJMA ** 2 * e2JMA
        jmaJMA[i] = e2JMA + prev_jmaJMA
        prev_jmaJMA = jmaJMA[i]
    
    # JMA signals
    if usejmaJMA:
        if usecolorjmaJMA:
            signalmaJMALong = (jmaJMA > np.concatenate([[jmaJMA[0]], jmaJMA[:-1]])) & (close > jmaJMA)
            signalmaJMAShort = (jmaJMA < np.concatenate([[jmaJMA[0]], jmaJMA[:-1]])) & (close < jmaJMA)
        else:
            signalmaJMALong = close > jmaJMA
            signalmaJMAShort = close < jmaJMA
    else:
        signalmaJMALong = np.ones(n, dtype=bool)
        signalmaJMAShort = np.ones(n, dtype=bool)
    
    finalLongSignalJMA = signalmaJMAShort if inverseJMA else signalmaJMALong
    finalShortSignalJMA = signalmaJMALong if inverseJMA else signalmaJMAShort
    
    # Trendilo calculations
    pct_change = np.zeros(n)
    for i in range(trendilo_smooth, n):
        pct_change[i] = (close[i] - close[i - trendilo_smooth]) / close[i - trendilo_smooth] * 100
    
    avg_pct_change = np.zeros(n)
    for i in range(trendilo_length - 1, n):
        window = pct_change[i - trendilo_length + 1:i + 1]
        m = trendilo_length
        offset_val = trendilo_offset * (m - 1)
        k = np.arange(m)
        if trendilo_sigma == 0:
            weights = np.ones(m) / m
        else:
            weights = np.exp(-(k - offset_val) ** 2 / (2 * (trendilo_sigma * 0.5) ** 2))
        weights = weights / np.sum(weights)
        avg_pct_change[i] = np.sum(window * weights)
    
    rms = np.zeros(n)
    for i in range(trendilo_length - 1, n):
        window = avg_pct_change[i - trendilo_length + 1:i + 1]
        rms[i] = trendilo_bmult * np.sqrt(np.mean(window ** 2))
    
    trendilo_dir = np.zeros(n)
    for i in range(n):
        if avg_pct_change[i] > rms[i]:
            trendilo_dir[i] = 1
        elif avg_pct_change[i] < -rms[i]:
            trendilo_dir[i] = -1
        else:
            trendilo_dir[i] = 0
    
    # Stiffness calculations
    boundStiffness = np.zeros(n)
    sumAboveStiffness = np.zeros(n)
    stiffness = np.zeros(n)
    
    sma = np.zeros(n)
    for i in range(maLengthStiffness - 1, n):
        sma[i] = np.mean(close[i - maLengthStiffness + 1:i + 1])
    
    std_dev = np.zeros(n)
    for i in range(maLengthStiffness - 1, n):
        std_dev[i] = np.std(close[i - maLengthStiffness + 1:i + 1], ddof=0)
    
    for i in range(n):
        boundStiffness[i] = sma[i] - 0.2 * std_dev[i]
    
    for i in range(stiffLength - 1, n):
        count = 0
        for j in range(stiffLength):
            if i - j >= 0 and close[i - j] > boundStiffness[i - j]:
                count += 1
        sumAboveStiffness[i] = count
    
    stiffness_raw = np.zeros(n)
    for i in range(n):
        if stiffLength > 0:
            stiffness_raw[i] = sumAboveStiffness[i] * 100 / stiffLength
    
    alpha_stiff = 2.0 / (stiffSmooth + 1)
    stiffness[stiffSmooth] = np.mean(stiffness_raw[:stiffSmooth + 1])
    for i in range(stiffSmooth + 1, n):
        stiffness[i] = alpha_stiff * stiffness_raw[i] + (1 - alpha_stiff) * stiffness[i - 1]
    
    signalStiffness = np.zeros(n)
    for i in range(n):
        if useStiffness:
            signalStiffness[i] = 1.0 if stiffness[i] > thresholdStiffness else 0.0
        else:
            signalStiffness[i] = 1.0
    
    # Entry conditions
    long_condition = finalLongSignalJMA & (trendilo_dir == 1) & (signalStiffness == 1)
    short_condition = finalShortSignalJMA & (trendilo_dir == -1) & (signalStiffness == -1)
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(n):
        if i > 0 and long_condition.iloc[i] if hasattr(long_condition, 'iloc') else long_condition[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = close[i]
            
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
        
        if i > 0 and short_condition.iloc[i] if hasattr(short_condition, 'iloc') else short_condition[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = close[i]
            
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
    
    return entries