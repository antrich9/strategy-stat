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
    # Constants from strategy inputs
    k = 3
    train_n = 252
    sdTapMult = 1.0
    atrLen = 14
    slAtrMult = 1.5
    tradeDir = "Both"
    
    close_prices = df['close']
    high_prices = df['high']
    low_prices = df['low']
    open_prices = df['open']
    
    # Compute ATR (Wilder method)
    high_low = high_prices - low_prices
    high_close = np.abs(high_prices - close_prices.shift(1))
    low_close = np.abs(low_prices - close_prices.shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0/atrLen, adjust=False).mean()
    
    # Engulfing patterns
    bullEngulf = (close_prices > high_prices.shift(1)) & (open_prices <= close_prices.shift(1))
    bearEngulf = (close_prices < low_prices.shift(1)) & (open_prices >= close_prices.shift(1))
    
    # K-Means clustering on training data (last bar in training set)
    # Using k clusters, train_n bars back
    train_end_idx = len(df) - 1
    train_start_idx = max(0, train_end_idx - train_n + 1)
    train_data = close_prices.iloc[train_start_idx:train_end_idx + 1].values
    
    # Initialize cluster centers
    clust_centers = []
    clust_centers.append(train_data[int(train_n * 0.02)])
    for i in range(1, k):
        clust_centers.append(train_data[int(train_n * (i / k))])
    
    clust_centers = np.array(clust_centers)
    
    # K-Means iteration
    max_iterations = 100
    for _ in range(max_iterations):
        # Assign points to clusters
        clusters = [[] for _ in range(k)]
        for price in train_data:
            distances = np.abs(clusters - price) if len(clusters.shape) > 1 else np.abs(clust_centers - price)
            idx = np.argmin(distances)
            clusters[idx].append(price)
        
        # Update centers
        new_centers = np.zeros(k)
        for i in range(k):
            if len(clusters[i]) > 0:
                new_centers[i] = np.mean(clusters[i])
            else:
                new_centers[i] = clust_centers[i]
        
        # Check convergence
        if np.sum((new_centers - clust_centers) ** 2) < 0.01:
            clust_centers = new_centers
            break
        clust_centers = new_centers
    
    # Compute SD for each cluster
    sd_clust = np.zeros(k)
    for i in range(k):
        if len(clusters[i]) > 0:
            sd_clust[i] = np.std(clusters[i], ddof=0)
        else:
            sd_clust[i] = 0
    
    # Tap detection helpers - build boolean series
    tapped_support = pd.Series(False, index=df.index)
    tapped_resistance = pd.Series(False, index=df.index)
    best_sup_level = pd.Series(np.nan, index=df.index)
    best_res_level = pd.Series(np.nan, index=df.index)
    
    for i in range(len(df)):
        low_val = low_prices.iloc[i]
        high_val = high_prices.iloc[i]
        close_val = close_prices.iloc[i]
        
        # Check support taps (long)
        for j in range(k):
            cPrice = clust_centers[j]
            cSD = sd_clust[j]
            tol = cSD * sdTapMult
            if low_val <= cPrice + tol and low_val >= cPrice - tol:
                tapped_support.iloc[i] = True
                best_sup_level.iloc[i] = cPrice
                break
        
        # Check resistance taps (short)
        for j in range(k):
            cPrice = clust_centers[j]
            cSD = sd_clust[j]
            tol = cSD * sdTapMult
            if high_val >= cPrice - tol and high_val <= cPrice + tol:
                tapped_resistance.iloc[i] = True
                best_res_level.iloc[i] = cPrice
                break
    
    # Entry conditions
    no_pos = True
    trade_num = 1
    entries = []
    
    for i in range(1, len(df)):
        if no_pos:
            # Long entries
            if tradeDir == "Long" or tradeDir == "Both":
                if tapped_support.iloc[i] and bullEngulf.iloc[i]:
                    if not np.isnan(best_sup_level.iloc[i]) and close_prices.iloc[i] >= best_sup_level.iloc[i]:
                        entry_ts = int(df['time'].iloc[i])
                        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                        entry_price = close_prices.iloc[i]
                        
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
                        no_pos = False
            
            # Short entries
            if no_pos and (tradeDir == "Short" or tradeDir == "Both"):
                if tapped_resistance.iloc[i] and bearEngulf.iloc[i]:
                    if not np.isnan(best_res_level.iloc[i]) and close_prices.iloc[i] <= best_res_level.iloc[i]:
                        entry_ts = int(df['time'].iloc[i])
                        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                        entry_price = close_prices.iloc[i]
                        
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
                        no_pos = False
        
        # Reset no_pos when exiting (simplified - would need to track exits properly)
        # For entry logic only, we assume positions don't overlap for simplicity
        no_pos = True
    
    return entries