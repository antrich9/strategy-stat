import pandas as pd
import numpy as np
from datetime import datetime, timezone

def alma(src, length, offset, sigma):
    m = int((length - 1) * offset)
    s = length / sigma
    weights = np.zeros(length)
    for i in range(length):
        weights[i] = np.exp(-((i - m) ** 2) / (2 * s * s))
    weights = weights / weights.sum()
    result = np.zeros(len(src))
    for i in range(length - 1, len(src)):
        result[i] = np.sum(weights * src.iloc[i - length + 1:i + 1].values * weights)
    return pd.Series(result, index=src.index)

def wilders_ema(series, length):
    alpha = 1.0 / length
    result = series.ewm(alpha=alpha, adjust=False).mean()
    return result

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close'].copy()
    open_ = df['open'].copy()
    high = df['high'].copy()
    low = df['low'].copy()
    
    almaLen = 40
    almaOffset = 0.85
    almaSigma = 6.0
    ffUpTol = 2.5
    ffDnTol = 2.5
    trLen = 50
    trOffset = 0.85
    trSigma = 6
    trBmult = 1.0
    adxLen = 14
    adxThr = 35
    chopLen = 14
    chopThr = 50.0
    
    almaVal = alma(close, almaLen, almaOffset, almaSigma)
    almaUp = close > almaVal
    almaDn = close < almaVal
    
    prevUpperBound = np.where(close.shift(1).values >= open_.shift(1).values, close.shift(1).values, open_.shift(1).values)
    prevLowerBound = np.where(close.shift(1).values <= open_.shift(1).values, close.shift(1).values, open_.shift(1).values)
    
    ffUpPct = np.where(high.values > prevUpperBound, 100.0 * (high.values - prevUpperBound) / prevUpperBound, 0.0)
    ffDnPct = np.where(low.values < prevLowerBound, 100.0 * (prevLowerBound - low.values) / prevLowerBound, 0.0)
    
    fatFingerUp = pd.Series(ffUpPct >= ffUpTol, index=df.index)
    fatFingerDown = pd.Series(ffDnPct >= ffDnTol, index=df.index)
    
    trPch = close.pct_change() * 100
    trAvpch = alma(trPch, trLen, trOffset, trSigma)
    trRMS = trBmult * np.sqrt((trAvpch ** 2).rolling(trLen).mean())
    trDir = pd.Series(np.where(trAvpch.values > trRMS.values, 1, np.where(trAvpch.values < -trRMS.values, -1, 0)), index=df.index)
    trendiloGreen = trDir == 1
    trendiloRed = trDir == -1
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = pd.Series(np.where((plus_dm.values > minus_dm.values) & (plus_dm.values > 0), plus_dm.values, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((minus_dm.values > plus_dm.values) & (minus_dm.values > 0), minus_dm.values, 0.0), index=df.index)
    atr_val = wilders_ema(high - low, adxLen)
    plus_di = 100 * wilders_ema(plus_dm, adxLen) / atr_val
    minus_di = 100 * wilders_ema(minus_dm, adxLen) / atr_val
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = wilders_ema(dx, adxLen)
    adxStrong = adx_val >= adxThr
    
    chop_hh = high.rolling(chopLen).max()
    chop_ll = low.rolling(chopLen).min()
    chop_range = chop_hh - chop_ll
    chop_atr_sum = atr_val.rolling(chopLen).sum()
    chop_index = pd.Series(np.where(chop_range.values > 0, 100 * np.log10(chop_atr_sum.values / chop_range.values) / np.log10(chopLen), 50.0), index=df.index)
    chop_trending = chop_index < chopThr
    
    timestamps = pd.to_datetime(df['time'], unit='s', utc=True)
    date_from = datetime(2020, 1, 1, tzinfo=timezone.utc)
    date_to = datetime(2099, 12, 31, tzinfo=timezone.utc)
    in_date_range = (timestamps >= date_from) & (timestamps <= date_to)
    
    long_cond = almaUp & trendiloGreen & adxStrong & fatFingerUp & chop_trending & in_date_range
    short_cond = almaDn & trendiloRed & adxStrong & fatFingerDown & chop_trending & in_date_range
    
    entries = []
    trade_num = 1
    position_size = 0
    
    for i in range(len(df)):
        if i == 0:
            continue
        if pd.isna(almaVal.iloc[i]) or pd.isna(trAvpch.iloc[i]) or pd.isna(adx_val.iloc[i]) or np.isnan(chop_index.iloc[i]):
            continue
        if position_size == 0:
            if long_cond.iloc[i]:
                entry_ts = int(df['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(close.iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(close.iloc[i]),
                    'raw_price_b': float(close.iloc[i])
                })
                trade_num += 1
                position_size = 1
            elif short_cond.iloc[i]:
                entry_ts = int(df['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(close.iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(close.iloc[i]),
                    'raw_price_b': float(close.iloc[i])
                })
                trade_num += 1
                position_size = -1
        elif position_size > 0 and short_cond.iloc[i]:
            position_size = 0
        elif position_size < 0 and long_cond.iloc[i]:
            position_size = 0
    
    return entries