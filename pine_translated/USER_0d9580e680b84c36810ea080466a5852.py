import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Parameters from Pine Script inputs
    left = 20
    right = 15
    atrLen = 30
    
    entries = []
    trade_num = 1
    
    n = len(df)
    min_bars = left + right + 1
    if n < min_bars:
        return entries
    
    # Calculate Heikin Ashi (src = "HA")
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
    ha_open = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = (df['open'].iloc[0] + ha_close.iloc[0]) / 2.0
    for i in range(1, n):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2.0
    
    ha_high = pd.concat([df['high'], ha_open], axis=1).max(axis=1)
    ha_low = pd.concat([df['low'], ha_open], axis=1).min(axis=1)
    
    # Calculate ATR (Wilder)
    high_low = df['high'] - df['low']
    high_close_prev = np.abs(df['high'] - df['close'].shift(1))
    low_close_prev = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0/atrLen, adjust=False).mean()
    
    # Detect pivots
    swing_high_idx = []
    swing_low_idx = []
    
    for i in range(left, n - right):
        # Swing high: highest in left bars, highest in right bars
        is_high = True
        for j in range(1, left + 1):
            if ha_high.iloc[i] <= ha_high.iloc[i-j]:
                is_high = False
                break
        if is_high:
            for j in range(1, right + 1):
                if ha_high.iloc[i] < ha_high.iloc[i+j]:
                    is_high = False
                    break
        if is_high:
            swing_high_idx.append(i)
        
        # Swing low: lowest in left bars, lowest in right bars
        is_low = True
        for j in range(1, left + 1):
            if ha_low.iloc[i] >= ha_low.iloc[i-j]:
                is_low = False
                break
        if is_low:
            for j in range(1, right + 1):
                if ha_low.iloc[i] > ha_low.iloc[i+j]:
                    is_low = False
                    break
        if is_low:
            swing_low_idx.append(i)
    
    # Entry logic based on detectBO (breakout) and detectBD (breakdown)
    last_signal_bar = -1
    
    for i in range(min_bars, n):
        if i < atrLen or pd.isna(atr.iloc[i]):
            continue
        
        # Get resistance (swing highs) and support (swing lows) levels
        recent_sh_idx = [idx for idx in swing_high_idx if idx < i and idx > last_signal_bar]
        recent_sl_idx = [idx for idx in swing_low_idx if idx < i and idx > last_signal_bar]
        
        resistance_levels = [ha_high.iloc[idx] for idx in recent_sh_idx]
        support_levels = [ha_low.iloc[idx] for idx in recent_sl_idx]
        
        resistance = max(resistance_levels) if resistance_levels else 0.0
        support = min(support_levels) if support_levels else 0.0
        
        close_price = df['close'].iloc[i]
        
        # Long entry on breakout (price breaks above resistance)
        if resistance > 0 and close_price > resistance:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': close_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': resistance,
                'raw_price_b': resistance
            })
            trade_num += 1
            last_signal_bar = i
        
        # Short entry on breakdown (price breaks below support)
        elif support > 0 and close_price < support:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': close_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': support,
                'raw_price_b': support
            })
            trade_num += 1
            last_signal_bar = i
    
    return entries