import pandas as pd
import numpy as np
from datetime import datetime, timezone

def calculate_wilder_rsi(series, length):
    """Wilder's RSI implementation"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_wilder_atr(high, low, close, length):
    """Wilder's ATR implementation"""
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.
    """
    length = 10
    minMult = 1.0
    maxMult = 5.0
    step = 0.5
    perfAlpha = 10.0
    fromCluster = 'Best'
    showSignals = True
    maxIter = 1000
    maxData = 10000
    
    hl2 = (df['high'] + df['low']) / 2
    
    atr = calculate_wilder_atr(df['high'], df['low'], df['close'], length)
    
    factors = np.arange(minMult, maxMult + step/2, step)
    n_factors = len(factors)
    
    perf_vals = np.zeros((n_factors, len(df)))
    
    for k, factor in enumerate(factors):
        upper = hl2.copy()
        lower = hl2.copy()
        trend = np.zeros(len(df), dtype=int)
        output = hl2.copy()
        perf = np.zeros(len(df))
        
        up = hl2 + atr * factor
        dn = hl2 - atr * factor
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > upper.iloc[i-1]:
                trend[i] = 1
            elif df['close'].iloc[i] < lower.iloc[i-1]:
                trend[i] = 0
            else:
                trend[i] = trend[i-1]
            
            upper.iloc[i] = up.iloc[i] if df['close'].iloc[i-1] < upper.iloc[i-1] else min(up.iloc[i], upper.iloc[i-1])
            lower.iloc[i] = dn.iloc[i] if df['close'].iloc[i-1] > lower.iloc[i-1] else max(dn.iloc[i], lower.iloc[i-1])
            
            output.iloc[i] = lower.iloc[i] if trend[i] == 1 else upper.iloc[i]
            
            diff = np.sign(df['close'].iloc[i-1] - output.iloc[i-1]) if not pd.isna(output.iloc[i-1]) else 0
            if pd.isna(perf[i-1]):
                perf[i] = (2/(perfAlpha+1)) * ((df['close'].iloc[i] - df['close'].iloc[i-1]) * diff)
            else:
                perf[i] = perf[i-1] + (2/(perfAlpha+1)) * ((df['close'].iloc[i] - df['close'].iloc[i-1]) * diff - perf[i-1])
        
        perf_vals[k] = perf
    
    clusterIndex = {'Best': 2, 'Average': 1, 'Worst': 0}[fromCluster]
    
    target_perf_arr = np.zeros(len(df))
    os = np.zeros(len(df), dtype=int)
    
    den = df['close'].ewm(span=int(perfAlpha), adjust=False).mean()
    prev_diff = df['close'].diff().abs()
    den = prev_diff.ewm(span=int(perfAlpha), adjust=False).mean()
    
    for i in range(1, len(df)):
        if i < 2:
            continue
        
        bar_idx = i
        if len(df) - bar_idx > maxData:
            target_perf_arr[i] = target_perf_arr[i-1] if not pd.isna(target_perf_arr[i-1]) else 0
            os[i] = os[i-1]
            continue
        
        start_idx = max(0, bar_idx - maxData)
        data_perfs = perf_vals[:, start_idx:bar_idx+1]
        
        if data_perfs.size == 0:
            target_perf_arr[i] = target_perf_arr[i-1] if not pd.isna(target_perf_arr[i-1]) else 0
            os[i] = os[i-1]
            continue
        
        flat_perfs = data_perfs.flatten()
        
        valid_mask = ~np.isnan(flat_perfs) & (flat_perfs != 0)
        if valid_mask.sum() < 3:
            target_perf_arr[i] = target_perf_arr[i-1] if not pd.isna(target_perf_arr[i-1]) else 0
            os[i] = os[i-1]
            continue
        
        valid_perfs = flat_perfs[valid_mask]
        
        sorted_perfs = np.sort(valid_perfs)
        q1_idx = int(len(sorted_perfs) * 0.25)
        q2_idx = int(len(sorted_perfs) * 0.50)
        q3_idx = int(len(sorted_perfs) * 0.75)
        
        if q1_idx >= len(sorted_perfs): q1_idx = len(sorted_perfs) - 1
        if q2_idx >= len(sorted_perfs): q2_idx = len(sorted_perfs) - 1
        if q3_idx >= len(sorted_perfs): q3_idx = len(sorted_perfs) - 1
        
        centroids = np.array([sorted_perfs[q1_idx], sorted_perfs[q2_idx], sorted_perfs[q3_idx]])
        
        perf_clusters = [[], [], []]
        factor_clusters = [[], []]
        
        for iter_num in range(maxIter):
            perf_clusters = [[], [], []]
            
            for j in range(len(valid_perfs)):
                dists = np.abs(valid_perfs[j] - centroids)
                idx = np.argmin(dists)
                perf_clusters[idx].append(valid_perfs[j])
            
            if all(len(c) == 0 for c in perf_clusters):
                break
            
            new_centroids = []
            for c in perf_clusters:
                if len(c) > 0:
                    new_centroids.append(np.mean(c))
                else:
                    new_centroids.append(centroids[len(new_centroids)])
            
            if len(new_centroids) < 3:
                while len(new_centroids) < 3:
                    new_centroids.append(centroids[len(new_centroids)] if len(new_centroids) < len(centroids) else 0)
            
            new_centroids = np.array(new_centroids[:3])
            
            if np.allclose(centroids, new_centroids, rtol=1e-5):
                centroids = new_centroids
                break
            
            centroids = new_centroids
        
        if clusterIndex < len(perf_clusters) and len(perf_clusters[clusterIndex]) > 0:
            target_perf_arr[i] = np.mean(perf_clusters[clusterIndex])
        else:
            target_perf_arr[i] = target_perf_arr[i-1] if not pd.isna(target_perf_arr[i-1]) else 0
        
        if pd.isna(den.iloc[i]) or den.iloc[i] == 0:
            target_perf_arr[i] = target_perf_arr[i-1] if not pd.isna(target_perf_arr[i-1]) else 0
            os[i] = os[i-1]
            continue
        
        perf_idx_val = max(target_perf_arr[i], 0) / den.iloc[i]
        
        up_val = hl2.iloc[i] + atr.iloc[i] * 3.0
        dn_val = hl2.iloc[i] - atr.iloc[i] * 3.0
        
        upper_ts = up_val if df['close'].iloc[i-1] >= target_perf_arr[i-1] else min(up_val, target_perf_arr[i-1]) if not pd.isna(target_perf_arr[i-1]) else up_val
        lower_ts = dn_val if df['close'].iloc[i-1] <= target_perf_arr[i-1] else max(dn_val, target_perf_arr[i-1]) if not pd.isna(target_perf_arr[i-1]) else dn_val
        
        if df['close'].iloc[i] > upper_ts:
            os[i] = 1
        elif df['close'].iloc[i] < lower_ts:
            os[i] = 0
        else:
            os[i] = os[i-1] if i > 0 else 0
    
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if pd.isna(atr.iloc[i]):
            continue
        
        den_val = den.iloc[i] if not pd.isna(den.iloc[i]) and den.iloc[i] != 0 else 1
        perf_idx_val = max(target_perf_arr[i], 0) / den_val
        
        signal_rating = perf_idx_val * 10
        
        if showSignals:
            if i > 0 and os.iloc[i] == 1 and os.iloc[i-1] == 0:
                ts_val = lower.iloc[i] if os[i] == 1 else upper.iloc[i]
                continue
        
        longCondition = (signal_rating >= 7) and (os.iloc[i] == 1)
        shortCondition = (signal_rating >= 7) and (os.iloc[i] == 0)
        
        if longCondition:
            entry_time_str = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': entry_time_str,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        
        if shortCondition:
            entry_time_str = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': entry_time_str,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries