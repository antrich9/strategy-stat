import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Default input parameters from the Pine Script
    PDCM = 70
    CDBA = 100
    FDB = 1.3
    ma_len_b = 8
    modo_tipo = 'CON FILTRADO DE TENDENCIA'
    VVEV = True
    VVER = True

    # Body and range
    body = (df['close'] - df['open']).abs()
    range_ = df['high'] - df['low']
    range_ = range_.replace(0, np.nan)

    body_pct = body * 100 / range_

    # Elephant candle base conditions
    VVE_0 = df['close'] > df['open']
    VRE_0 = df['close'] < df['open']

    VVE_1 = VVE_0 & (body_pct >= PDCM)
    VRE_1 = VRE_0 & (body_pct >= PDCM)

    # Wilder ATR
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / CDBA, adjust=False).mean()

    VVE_2 = VVE_1 & (body >= atr.shift(1) * FDB)
    VRE_2 = VRE_1 & (body >= atr.shift(1) * FDB)

    # Fast moving average (SMA) and its direction
    ma_fast = close.rolling(ma_len_b).mean()
    direction_b_gt0 = ma_fast > ma_fast.shift(1)
    direction_b_lt0 = ma_fast < ma_fast.shift(1)

    # Apply trend filter based on mode
    if modo_tipo == 'CON FILTRADO DE TENDENCIA':
        VVE_3 = VVE_2 & direction_b_gt0
        VRE_3 = VRE_2 & direction_b_lt0
    else:
        VVE_3 = VVE_2
        VRE_3 = VRE_2

    # Final entry signals (VVEV/VVER are always True in this configuration)
    long_entry = VVE_3
    short_entry = VRE_3

    # Generate entry list
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if long_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries