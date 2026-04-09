import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    donchLength = 20
    PDCM = 70
    CDBA = 100
    FDB = 1.3
    VVEV = True
    VVER = True
    modo_tipo = "CON FILTRADO DE TENDENCIA"
    config_tend_alc = "DIRECCION MEDIA RAPIDA ALCISTA"
    config_tend_baj = "DIRECCION MEDIA RAPIDA BAJISTA"
    ma_type = "SMA"
    ma_len = 20
    ma_type_b = "SMA"
    ma_len_b = 8
    reaction_ma_1 = 1
    reaction_ma_2 = 1

    high = df['high']
    low = df['low']
    close = df['close']
    open_price = df['open']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=CDBA).mean()

    VVE_0 = close > open_price
    VRE_0 = close < open_price

    body = (open_price - close).abs()
    range_bar = high - low
    body_pct = body * 100 / range_bar

    VVE_1 = VVE_0 & (body_pct >= PDCM)
    VRE_1 = VRE_0 & (body_pct >= PDCM)

    VVE_2 = VVE_1 & (body >= atr.shift(1) * FDB)
    VRE_2 = VRE_1 & (body >= atr.shift(1) * FDB)

    def calc_ma(src, length, ma_type):
        if ma_type == "SMA":
            return src.rolling(length).mean()
        elif ma_type == "EMA":
            return src.ewm(span=length, adjust=False).mean()
        elif ma_type == "WMA":
            weights = np.arange(1, length + 1)
            return src.rolling(length).apply(lambda x: np.dot(x, weights[:len(x)]) / weights[:len(x)].sum(), raw=True)
        elif ma_type == "VWMA":
            vol = df['volume']
            return (src * vol).rolling(length).sum() / vol.rolling(length).sum()
        elif ma_type == "SMMA":
            return src.ewm(alpha=1.0/length, adjust=False).mean()
        elif ma_type == "DEMA":
            ema1 = src.ewm(span=length, adjust=False).mean()
            return 2 * ema1 - ema1.ewm(span=length, adjust=False).mean()
        elif ma_type == "TEMA":
            ema1 = src.ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            return 3 * (ema1 - ema2) + ema2.ewm(span=length, adjust=False).mean()
        elif ma_type == "HullMA":
            hull_len = int(np.sqrt(length))
            wma1 = src.rolling(length // 2).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / (len(x) * (len(x) + 1) / 2), raw=True)
            wma2 = src.rolling(length).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / (len(x) * (len(x) + 1) / 2), raw=True)
            return (2 * wma1 - wma2).rolling(hull_len).mean()
        elif ma_type == "SSMA":
            a1 = np.exp(-1.414 * np.pi / length)
            b1 = 2 * a1 * np.cos(1.414 * np.pi / length)
            c2 = b1
            c3 = -a1 * a1
            c1 = 1 - c2 - c3
            sma_vals = src.rolling(length).mean()
            result = pd.Series(0.0, index=src.index)
            result.iloc[length-1] = sma_vals.iloc[length-1]
            for i in range(length, len(src)):
                result.iloc[i] = c1 * (src.iloc[i] + src.iloc[i-1]) / 2 + c2 * result.iloc[i-1] + c3 * result.iloc[i-2]
            return result
        elif ma_type == "ZEMA":
            ema1 = src.ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            return ema1 + (ema1 - ema2)
        elif ma_type == "TMA":
            return src.rolling(length).mean().rolling(length).mean()
        else:
            return src.rolling(length).mean()

    ma_series = calc_ma(close, ma_len, ma_type)
    ma_series_b = calc_ma(close, ma_len_b, ma_type_b)

    ma_series = ma_series.fillna(0)
    ma_series_b = ma_series_b.fillna(0)

    direction = pd.Series(0.0, index=close.index)
    rising_count = 0
    falling_count = 0
    for i in range(1, len(close)):
        if ma_series.iloc[i] > ma_series.iloc[i-1]:
            rising_count += 1
            falling_count = 0
        elif ma_series.iloc[i] < ma_series.iloc[i-1]:
            falling_count += 1
            rising_count = 0
        else:
            rising_count = 0
            falling