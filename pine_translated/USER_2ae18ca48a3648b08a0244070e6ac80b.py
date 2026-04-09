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
    open_prices = df['open']
    
    # E2PSS Parameters
    useE2PSS = True
    inverseE2PSS = False
    PeriodE2PSS = 15
    PriceE2PSS = (df['high'] + df['low']) / 2
    
    # E2PSS Calculation
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * np.pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    Filt2 = np.zeros(len(df))
    TriggerE2PSS = np.zeros(len(df))
    Filt2[0] = PriceE2PSS.iloc[0]
    Filt2[1] = PriceE2PSS.iloc[1]
    
    for i in range(2, len(df)):
        if i < 3:
            Filt2[i] = PriceE2PSS.iloc[i]
        else:
            Filt2[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
    
    TriggerE2PSS = pd.Series(Filt2).shift(1).fillna(method='bfill').values
    
    signalLongE2PSS = Filt2 > TriggerE2PSS if useE2PSS else np.ones(len(df), dtype=bool)
    signalShortE2PSS = Filt2 < TriggerE2PSS if useE2PSS else np.ones(len(df), dtype=bool)
    
    signalLongE2PSSFinal = signalShortE2PSS if inverseE2PSS else signalLongE2PSS
    signalShortE2PSSFinal = signalLongE2PSS if inverseE2PSS else signalShortE2PSS
    
    # Trendilo Parameters
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    # Trendilo Calculation
    close_shifted = close.shift(trendilo_smooth)
    pct_change = (close - close_shifted) / close * 100
    pct_change = pct_change.fillna(0)
    
    # ALMA implementation
    def alma(series, length, offset, sigma):
        m = (offset * (length - 1))
        s = series.rolling(length).sum() / length
        w = np.exp(-((np.arange(length) - m) ** 2) / (2 * sigma ** 2))
        w = w / w.sum()
        result = series.rolling(length).apply(lambda x: np.dot(x, w), raw=True)
        return result
    
    avg_pct_change = alma(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    avg_pct_change = avg_pct_change.fillna(0)
    
    rms = trendilo_bmult * np.sqrt((avg_pct_change ** 2).rolling(trendilo_length).sum() / trendilo_length)
    
    trendilo_dir = np.where(avg_pct_change > rms, 1, np.where(avg_pct_change < -rms, -1, 0))
    
    # Stiffness Parameters
    useStiffness = False
    maLengthStiffness = 100
    stiffLength = 60
    stiffSmooth = 3
    thresholdStiffness = 90
    
    # Stiffness Calculation
    boundStiffness = close.rolling(maLengthStiffness).mean() - 0.2 * close.rolling(maLengthStiffness).std()
    sumAboveStiffness = (close > boundStiffness).rolling(stiffLength).sum()
    stiffness = (sumAboveStiffness * 100 / stiffLength).ewm(span=stiffSmooth, adjust=False).mean()
    
    signalStiffness = stiffness > thresholdStiffness if useStiffness else np.ones(len(df), dtype=bool)
    
    # Entry Conditions
    long_condition = signalLongE2PSSFinal & (trendilo_dir == 1) & signalStiffness
    short_condition = signalShortE2PSSFinal & (trendilo_dir == -1) & signalStiffness
    
    # Generate Entries
    entries = []
    trade_num = 1
    in_position = False
    current_direction = None
    
    for i in range(len(df)):
        if in_position:
            continue
        
        entry_price = close.iloc[i]
        
        if long_condition.iloc[i]:
            entry_ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
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
            in_position = True
            current_direction = 'long'
        elif short_condition.iloc[i]:
            entry_ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
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
            in_position = True
            current_direction = 'short'
    
    return entries