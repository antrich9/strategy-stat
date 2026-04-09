import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    length = 130
    coef = 0.2
    vcoef = 2.5
    signalLength = 5
    smoothVFI = False
    donchLength = 20
    PDCM = 70
    CDBA = 100
    FDB = 1.3
    ma_type = 'SMA'
    ma_len = 20
    ma_src_col = 'close'
    reaction_ma_1 = 1
    ma_type_b = 'SMA'
    ma_len_b = 8
    ma_src_b_col = 'close'
    reaction_ma_2 = 1
    modo_tipo = 'CON FILTRADO DE TENDENCIA'
    VVEV = True
    VVER = True
    config_tend_alc = 'DIRECCION MEDIA RAPIDA ALCISTA'
    config_tend_baj = 'DIRECCION MEDIA RAPIDA BAJISTA'

    typical = (df['high'] + df['low'] + df['close']) / 3.0
    inter = np.log(typical) - np.log(typical.shift(1))
    vinter = inter.rolling(window=30).std()
    cutoff = coef * vinter * df['close']
    vave = df['volume'].rolling(window=length).mean().shift(1)
    vmax = vave * vcoef
    vc = np.minimum(df['volume'], vmax)
    mf = typical - typical.shift(1)
    vcp = np.where(mf > cutoff, vc, np.where(mf < -cutoff, -vc, 0))
    vfi_ma_data = pd.Series(vcp).rolling(window=length).sum() / vave
    vfi = vfi_ma_data.rolling(window=3).mean() if smoothVFI else vfi_ma_data
    vfima = vfi.ewm(span=signalLength, adjust=False).mean()

    highestHigh = df['high'].rolling(window=donchLength).max()
    lowestLow = df['low'].rolling(window=donchLength).min()
    middleBand = (highestHigh + lowestLow) / 2

    body = np.abs(df['close'] - df['open'])
    full_range = df['high'] - df['low']
    body_pct = np.where(full_range != 0, (body / full_range) * 100, 0)

    def calculate_wilder_atr(data, period):
        tr1 = data['high'] - data['low']
        tr2 = np.abs(data['high'] - data['close'].shift(1))
        tr3 = np.abs(data['low'] - data['close'].shift(1))
        tr = pd.Series(np.maximum(np.maximum(tr1, tr2), tr3))
        atr = pd.Series(index=data.index, dtype=float)
        atr.iloc[period-1] = tr.iloc[:period].mean()
        alpha = 1.0 / period
        for i in range(period, len(tr)):
            atr.iloc[i] = alpha * tr.iloc[i] + (1 - alpha) * atr.iloc[i-1]
        return atr

    atr = calculate_wilder_atr(df, CDBA)
    atr_prev = atr.shift(1)

    VVE_0 = df['close'] > df['open']
    VRE_0 = df['close'] < df['open']
    VVE_1 = VVE_0 & (body_pct >= PDCM)
    VRE_1 = VRE_0 & (body_pct >= PDCM)
    VVE_2 = VVE_1 & (body >= atr_prev * FDB)
    VRE_2 = VRE_1 & (body >= atr_prev * FDB)

    def calculate_variant_ma(src, length, ma_type):
        if ma_type == 'EMA':
            return src.ewm(span=length, adjust=False).mean()
        elif ma_type == 'SMA':
            return src.rolling(window=length).mean()
        elif ma_type == 'WMA':
            weights = np.arange(1, length + 1)
            return src.rolling(window=length).apply(lambda x: (weights * x).sum() / weights.sum(), raw=True)
        elif ma_type == 'VWMA':
            return (src * df['volume']).rolling(window=length).sum() / df['volume'].rolling(window=length).sum()
        elif ma_type == 'SMMA':
            result = pd.Series(index=src.index, dtype=float)
            result.iloc[length-1] = src.iloc[:length].mean()
            for i in range(length, len(src)):
                result.iloc[i] = (result.iloc[i-1] * (length - 1) + src.iloc[i]) / length
            return result
        elif ma_type == 'DEMA':
            ema1 = src.ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            return 2 * ema1 - ema2
        elif ma_type == 'TEMA':
            ema1 = src.ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            ema3 = ema2.ewm(span=length, adjust=False).mean()
            return 3 * ema1 - 3 * ema2 + ema3
        elif ma_type == 'HullMA':
            wma_half = src.rolling(window=int(length/2)).apply(lambda x: (np.arange(1, len(x)+1) * x).sum() / np.arange(1, len(x)+1).sum(), raw=True)
            wma_full = src.rolling(window=length).apply(lambda x: (np.arange(1, len(x)+1) * x).sum() / np.arange(1, len(x)+1).sum(), raw=True)
            hull = 2 * wma_half - wma_full
            sqrt_len = int(np.round(np.sqrt(length)))
            return hull.rolling(window=sqrt_len).apply(lambda x: (np.arange(1, len(x)+1) * x).sum() / np.arange(1, len(x)+1).sum(), raw=True)
        elif ma_type == 'SSMA':
            a1 = np.exp(-1.414 * 3.14159 / length)
            b1 = 2 * a1 * np.cos(1.414 * 3.14159 / length)
            c2 = b1
            c3 = -a1 * a1
            c1 = 1 - c2 - c3
            result = pd.Series(index=src.index, dtype=float)
            v9_prev2 = 0.0
            v9_prev1 = 0.0
            for i in range(len(src)):
                if i == 0:
                    v9 = c1 * (src.iloc[i] + src.iloc[i]) / 2 + c2 * 0 + c3 * 0
                elif i == 1:
                    v9 = c1 * (src.iloc[i] + src.iloc[i-1]) / 2 + c2 * v9_prev1 + c3 * v9_prev2
                else:
                    v9 = c1 * (src.iloc[i] + src.iloc[i-1]) / 2 + c2 * v9_prev1 + c3 * v9_prev2
                v9_prev2 = v9_prev1
                v9_prev1 = v9
                result.iloc[i] = v9
            return result
        elif ma_type == 'ZEMA':
            ema1 = src.ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            return ema1 + ema1 - ema2
        elif ma_type == 'TMA':
            sma1 = src.rolling(window=length).mean()
            return sma1.rolling(window=length).mean()
        else:
            return src.rolling(window=length).mean()

    ma_series = calculate_variant_ma(df[ma_src_col], ma_len, ma_type)
    ma_series_b = calculate_variant_ma(df[ma_src_b_col], ma_len_b, ma_type_b)

    direction = pd.Series(0, index=df.index)
    direction.iloc[reaction_ma_1:] = np.where(
        ma_series.iloc[reaction_ma_1:].values > ma_series.shift(reaction_ma_1).iloc[reaction_ma_1:].values,
        1,
        np.where(
            ma_series.iloc[reaction_ma_1:] < ma_series.shift(reaction_ma_1).iloc[reaction_ma_1:].values,
            -1,
            0
        )
    )
    direction = direction.replace(0, np.nan).ffill().fillna(0)

    direction_b = pd.Series(0, index=df.index)
    direction_b.iloc[reaction_ma_2:] = np.where(
        ma_series_b.iloc[reaction_ma_2:].values > ma_series_b.shift(reaction_ma_2).iloc[reaction_ma_2:].values,
        1,
        np.where(
            ma_series_b.iloc[reaction_ma_2:] < ma_series_b.shift(reaction_ma_2).iloc[reaction_ma_2:].values,
            -1,
            0
        )
    )
    direction_b = direction_b.replace(0, np.nan).ffill().fillna(0)

    bull_conditions = {
        'PRECIO MAYOR A MEDIA RAPIDA': df['close'] > ma_series_b,
        'PRECIO MAYOR A MEDIA LENTA': df['close'] > ma_series,
        'PRECIO MAYOR A MEDIA RAPIDA Y LENTA': (df['close'] > ma_series) & (df['close'] > ma_series_b),
        'PRECIO MAYOR A MEDIA LENTA Y DIRECCION ALCISTA': (df['close'] > ma_series) & (direction > 0),
        'PRECIO MAYOR A MEDIA RAPIDA Y DIRECCION ALCISTA': (df['close'] > ma_series_b) & (direction_b > 0),
        'PRECIO MAYOR A MEDIA LENTA/RAPIDA Y DIRECCION ALCISTA': (df['close'] > ma_series) & (df['close'] > ma_series_b) & (direction > 0) & (direction_b > 0),
        'DIRECCION MEDIA LENTA ALCISTA': direction > 0,
        'DIRECCION MEDIA RAPIDA ALCISTA': direction_b > 0,
        'DIRECCION MEDIA LENTA/RAPIDA ALCISTA': (direction > 0) & (direction_b > 0),
        'NINGUNA CONDICION': pd.Series(True, index=df.index)
    }
    bear_conditions = {
        'PRECIO MENOR A MEDIA RAPIDA': df['close'] < ma_series_b,
        'PRECIO MENOR A MEDIA LENTA': df['close'] < ma_series,
        'PRECIO MENOR A MEDIA RAPIDA Y LENTA': (df['close'] < ma_series) & (df['close'] < ma_series_b),
        'PRECIO MENOR A MEDIA LENTA Y DIRECCION BAJISTA': (df['close'] < ma_series) & (direction < 0),
        'PRECIO MENOR A MEDIA RAPIDA Y DIRECCION BAJISTA': (df['close'] < ma_series_b) & (direction_b < 0),
        'PRECIO MENOR A MEDIA LENTA/RAPIDA Y DIRECCION BAJISTA': (df['close'] < ma_series) & (df['close'] < ma_series_b) & (direction < 0) & (direction_b < 0),
        'DIRECCION MEDIA LENTA BAJISTA': direction < 0,
        'DIRECCION MEDIA RAPIDA BAJISTA': direction_b < 0,
        'DIRECCION MEDIA LENTA/RAPIDA BAJISTA': (direction < 0) & (direction_b < 0),
        'NINGUNA CONDICION': pd.Series(True, index=df.index)
    }

    bull_cond = bull_conditions.get(config_tend_alc, pd.Series(True, index=df.index))
    bear_cond = bear_conditions.get(config_tend_baj, pd.Series(True, index=df.index))

    VVE_3 = VVE_2 & bull_cond
    VRE_3 = VRE_2 & bear_cond

    use_filter = modo_tipo == 'CON FILTRADO DE TENDENCIA'
    RES_VVE = (VVE_3 if use_filter else VVE_2) & VVEV
    RES_VRE = (VRE_3 if use_filter else VRE_2) & VVER

    entries = []
    trade_num = 1
    for i in range(len(df)):
        if RES_VVE.iloc[i]:
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
        if RES_VRE.iloc[i]:
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