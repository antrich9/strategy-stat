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
    
    # Parameters
    length = 20
    multiplier = 1.5
    volMultiplier = 2
    volumeSpikeMultiplier = 1.5
    priorTrendLength = 50
    start_ts = int(datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime(2020, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp())
    
    # Calculate Wilder ATR (True Range -> Wilder smoothing)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    
    # Volume MA (SMA)
    volMA = df['volume'].rolling(length).mean()
    
    # Prior Trend MA (SMA)
    maPriorTrend = df['close'].rolling(priorTrendLength).mean()
    
    # Minervini Criteria
    sma150 = df['close'].rolling(150).mean()
    sma200 = df['close'].rolling(200).mean()
    priceAboveMA150 = df['close'] > sma150
    priceAboveMA200 = df['close'] > sma200
    ma150AboveMA200 = sma150 > sma200
    highest52 = df['close'].rolling(52).max()
    priceWithin25PercentOfHigh = df['close'] > highest52 * 0.75
    minerviniCriteria = priceAboveMA150 & priceAboveMA200 & ma150AboveMA200 & priceWithin25PercentOfHigh
    
    # Volume Spike
    volumeSpike = df['volume'] > volMA * volumeSpikeMultiplier
    
    # Prior Uptrend
    priorUptrend = df['close'] > maPriorTrend
    
    # VCP Pattern
    contractingPrice = (df['low'] > df['low'].shift(1)) & (df['high'] < df['high'].shift(1)) & (atr < atr.shift(1) * multiplier)
    contractingVolume = df['volume'] < volMA * volMultiplier
    vcpPattern = contractingPrice & contractingVolume
    
    # Buy Point calculation (var float, persists until next vcpPattern)
    buyPoint = pd.Series(np.nan, index=df.index)
    last_buy_point = np.nan
    
    for i in range(len(df)):
        if vcpPattern.iloc[i]:
            last_buy_point = df['high'].iloc[i] + atr.iloc[i] * 0.1
        buyPoint.iloc[i] = last_buy_point
    
    # Entry loop
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(buyPoint.iloc[i]):
            continue
        if pd.isna(minerviniCriteria.iloc[i]) or pd.isna(volumeSpike.iloc[i]) or pd.isna(priorUptrend.iloc[i]):
            continue
        if not (minerviniCriteria.iloc[i] and volumeSpike.iloc[i] and priorUptrend.iloc[i]):
            continue
        ts = df['time'].iloc[i]
        if not (ts >= start_ts and ts <= end_ts):
            continue
        
        entry_price = df['close'].iloc[i]
        entries.append({
            'trade_num': trade_num,
            'direction': 'long',
            'entry_ts': int(ts),
            'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            'entry_price_guess': entry_price,
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': entry_price,
            'raw_price_b': entry_price
        })
        trade_num += 1
    
    return entries