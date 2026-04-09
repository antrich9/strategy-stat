import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # ------------------------------------------------------------------
    # Prepare basic series
    # ------------------------------------------------------------------
    close   = df['close']
    high    = df['high']
    low     = df['low']
    open_   = df['open']
    volume  = df['volume']

    # ------------------------------------------------------------------
    # 1. Trading‑window (London 07:45‑09:45 and 14:45‑16:45)
    # ------------------------------------------------------------------
    df['hour']   = df['time'].apply(lambda ts: datetime.utcfromtimestamp(ts).hour)
    df['minute'] = df['time'].apply(lambda ts: datetime.utcfromtimestamp(ts).minute)

    morning = (
        ((df['hour'] == 7) & (df['minute'] >= 45)) |
        (df['hour'] == 8) |
        ((df['hour'] == 9) & (df['minute'] <= 45))
    )
    afternoon = (
        ((df['hour'] == 14) & (df['minute'] >= 45)) |
        (df['hour'] == 15) |
        ((df['hour'] == 16) & (df['minute'] <= 45))
    )
    in_trading_window = morning | afternoon

    # ------------------------------------------------------------------
    # 2. Previous‑day high / low (daily high/low shifted by one day)
    # ------------------------------------------------------------------
    df['day'] = pd.to_datetime(df['time'], unit='s').dt.date
    daily_high = df.groupby('day')['high'].max()
    daily_low  = df.groupby('day')['low'].min()
    prev_day_high = df['day'].map(daily_high.shift(1))
    prev_day_low  = df['day'].map(daily_low.shift(1))

    # ------------------------------------------------------------------
    # 3. Volume filter (volfilt)
    # ------------------------------------------------------------------
    sma9   = volume.rolling(9).mean()
    volfilt = volume.shift(1) > sma9 * 1.5

    # ------------------------------------------------------------------
    # 4. ATR filter (atrfilt) – 20‑period Wilder ATR
    # ------------------------------------------------------------------
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False).mean()

    atrfilt = (low - high.shift(2) > atr) | (low.shift(2) - high > atr)

    # ------------------------------------------------------------------
    # 5. Trend filter (SMA 54) – disabled by default
    # ------------------------------------------------------------------
    sma54  = close.rolling(54).mean()
    loc2   = sma54 > sma54.shift(1)
    # default inputs have trend filter OFF → both pass always
    locfiltb = True   # bullish trend filter
    locfilts = True   # bearish trend filter

    # ------------------------------------------------------------------
    # 6. Bullish / Bearish Fair‑Value