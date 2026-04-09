import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def wilder_atr(high, low, close, length):
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    
    # Get daily data for trend
    daily = df.resample('D', on='time').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    isBullishDaily = daily['close'] > daily['open']
    isBearishDaily = daily['close'] < daily['open']
    
    # Calculate indicators
    atr14 = wilder_atr(df['high'], df['low'], df['close'], 14)
    atr20 = wilder_atr(df['high'], df['low'], df['close'], 20)
    sma54 = df['close'].rolling(54).mean()
    sma54_prev = sma54.shift(1)
    vol_sma9 = df['volume'].rolling(9).mean()
    
    # Filters
    volfilt = (df['volume'].shift(1) > vol_sma9.shift(1) * 1.5)
    atrfilt = ((df['low'] - df['high'].shift(2) > atr20 / 1.5) | (df['high'] - df['low'].shift(2) > atr20 / 1.5))
    loc2 = sma54 > sma54_prev
    locfiltb = loc2
    locfilts = ~loc2
    
    # FVG detection
    bull_fvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    bear_fvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfilts
    
    # Align daily trend with 15min data
    daily_reindexed = daily.reindex(df.index, method='ffill')
    isBullishDaily_aligned = daily_reindexed['close'] > daily_reindexed['open']
    isBearishDaily_aligned = daily_reindexed['close'] < daily_reindexed['open']
    
    # Entry conditions
    long_cond = bull_fvg & isBullishDaily_aligned
    short_cond = bear_fvg & isBearishDaily_aligned
    
    entries = []
    trade_num = 1
    
    for i in range(2, len(df)):
        if long_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries