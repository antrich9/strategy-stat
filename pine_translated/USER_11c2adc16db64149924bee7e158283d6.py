import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Inputs (defaults)
    length_atr = 100
    pdcm_threshold = 70.0
    fdb_multiplier = 1.3
    ma_len_fast = 8
    ma_len_slow = 20 # not used in default trend logic
    # ...

    # Calculate indicators
    # 1. ATR (Wilder)
    high = df['high']
    low = df['low']
    close = df['close']
    open_price = df['open']
    
    # Wilder ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder smoothing: EMA with alpha = 1/length
    atr = tr.ewm(alpha=1/length_atr, adjust=False).mean()
    
    # 2. Fast MA (SMA default)
    ma_fast = close.rolling(ma_len_fast).mean()
    
    # 3. Direction of Fast MA
    # direction_b := ta.rising(ma_series_b, reaction_ma_2) ? 1 : ta.falling(ma_series_b, reaction_ma_2) ? -1 : nz(direction_b[1])
    # reaction_ma_2 = 1
    direction_fast = pd.Series(0.0, index=close.index)
    # First bar with valid ma_fast
    for i in range(1, len(close)):
        if pd.isna(ma_fast.iloc[i]) or pd.isna(ma_fast.iloc[i-1]):
            continue
        if ma_fast.iloc[i] > ma_fast.iloc[i-1]:
            direction_fast.iloc[i] = 1
        elif ma_fast.iloc[i] < ma_fast.iloc[i-1]:
            direction_fast.iloc[i] = -1
        else:
            direction_fast.iloc[i] = direction_fast.iloc[i-1]
    
    # Alternatively, use a simpler approach with shift and fill
    # Actually, the loop is fine for clarity given the conditional logic.
    
    # Calculate conditions
    body = abs(open_price - close)
    body_pct = body * 100 / (high - low)
    
    # VVE_0: close > open
    vve_0 = close > open_price
    # VRE_0: close < open
    vre_0 = close < open_price
    
    # VVE_1: VVE_0 and body_pct >= pdcm
    vve_1 = vve_0 & (body_pct >= pdcm_threshold)
    vre_1 = vre_0 & (body_pct >= pdcm_threshold)
    
    # VVE_2: VVE_1 and body >= atr.shift(1) * fdb
    vve_2 = vve_1 & (body >= atr.shift(1) * fdb_multiplier)
    vre_2 = vre_1 & (body >= atr.shift(1) * fdb_multiplier)
    
    # VVE_3: VVE_2 and direction_fast > 0 (default config_tend_alc)
    vve_3 = vve_2 & (direction_fast > 0)
    # VRE_3: VRE_2 and direction_fast < 0 (default config_tend_baj)
    vre_3 = vre_2 & (direction_fast < 0)
    
    # Final signals (with defaults VVEV=True, modo_tipo='CON FILTRADO...')
    # If modo_tipo was 'SIN FILTRADO', it would be VVE_2, but defaults are CON FILTRADO.
    long_signal = vve_3
    short_signal = vre_3
    
    # Iterate and create entries
    entries = []
    trade_num = 1
    for i in range(len(df)):
        ts = int(df['time'].iloc[i])
        price = df['close'].iloc[i]
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        if long_signal.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
        elif short_signal.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
            
    return entries