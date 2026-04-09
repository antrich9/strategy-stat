import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Parameters from Pine Script
    length1 = 20
    mult = 2.0
    lengthKC = 20
    multKC = 1.5
    useTrueRange = True
    tetherLength = 50
    mg_length = 14
    period_ = 14
    
    high = df['high']
    low = df['low']
    close = df['close']
    time_col = df['time']
    
    # BB calculation
    basis = close.rolling(length1).mean()
    dev = multKC * close.rolling(length1).std()
    upperBB = basis + dev
    lowerBB = basis - dev
    
    # KC calculation
    ma = close.rolling(lengthKC).mean()
    if useTrueRange:
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        priceRange = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    else:
        priceRange = high - low
    
    rangeMa = priceRange.rolling(lengthKC).mean()
    upperKC = ma + rangeMa * multKC
    lowerKC = ma - rangeMa * multKC
    
    # Squeeze conditions
    sqzOn = (lowerBB > lowerKC) & (upperBB < upperKC)
    sqzOff = (lowerBB < lowerKC) & (upperBB > upperKC)
    
    # Linreg calculation for val
    # val = ta.linreg(source - ((highest + lowest)/2 + sma(close))/2, lengthKC, 0)
    highest_high = high.rolling(lengthKC).max()
    lowest_low = low.rolling(lengthKC).min()
    middle = (highest_high + lowest_low) / 2
    sma_close = close.rolling(lengthKC).mean()
    linreg_source = close - (middle + sma_close) / 2
    
    # Implement ta.linreg manually
    def rolling_linreg(series, length):
        result = pd.Series(np.nan, index=series.index)
        x = np.arange(length)
        x_mean = x.mean()
        sum_xx = ((x - x_mean) ** 2).sum()
        
        for i in range(length - 1, len(series)):
            y = series.iloc[i - length + 1:i + 1].values
            y_mean = y.mean()
            numerator = ((x - x_mean) * (y - y_mean)).sum()
            if sum_xx != 0:
                slope = numerator / sum_xx
                intercept = y_mean - slope * x_mean
                # offset=0 means value at most recent bar (index length-1)
                result.iloc[i] = intercept + slope * (length - 1)
        return result
    
    val = rolling_linreg(linreg_source, lengthKC)
    
    # McGinley Dynamic
    mg = pd.Series(np.nan, index=close.index)
    ema = close.ewm(span=mg_length, adjust=False).mean()
    
    # Find first valid index
    first_valid_idx = 0
    for idx in range(len(close)):
        if not pd.isna(close.iloc[idx]):
            first_valid_idx = idx
            break
    
    if first_valid_idx < len(close):
        mg.iloc[first_valid_idx] = ema.iloc[first_valid_idx]
        
        for i in range(first_valid_idx + 1, len(close)):
            prev_mg = mg.iloc[i-1]
            curr_close = close.iloc[i]
            if prev_mg != 0 and not pd.isna(prev_mg):
                mg.iloc[i] = prev_mg + (curr_close - prev_mg) / (mg_length * pow(curr_close / prev_mg, 4))
            else:
                mg.iloc[i] = ema.iloc[i]
    
    # Vortex Indicator
    VMP = (high - low.shift(1)).abs().rolling(period_).sum()
    VMM = (low - high.shift(1)).abs().rolling(period_).sum()
    
    # ATR(1) for Vortex
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    STR = tr.rolling(period_).sum()
    
    VIP = VMP / STR
    VIM = VMM / STR
    
    # Entry condition
    # longCondition = sqzOff and val > 0 and (VIP > VIM) and (close > mg)
    longCondition = sqzOff & (val > 0) & (VIP > VIM) & (close > mg)
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if longCondition.iloc[i]:
            entry_ts = int(time_col.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
            
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
    
    return entries