import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')
    
    # Wilder RSI
    def wilder_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    # Wilder ATR
    def wilder_atr(high, low, close, period):
        tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    atr_144 = wilder_atr(df['high'], df['low'], df['close'], 144)
    atr_14 = wilder_atr(df['high'], df['low'], df['close'], 14)
    fvgTH = 0.5
    
    # Previous day high/low
    daily = df.resample('D', on='datetime').agg({'high': 'max', 'low': 'min'})
    daily['prev_day_high'] = daily['high'].shift(1)
    daily['prev_day_low'] = daily['low'].shift(1)
    daily = daily.reset_index()
    df_daily = df[['datetime']].copy()
    df_daily['date'] = df_daily['datetime'].dt.date
    daily['date'] = daily['datetime'].dt.date
    df = df.merge(daily[['date', 'prev_day_high', 'prev_day_low']], on='date', how='left')
    df['prev_day_high'] = df['prev_day_high'].fillna(method='ffill')
    df['prev_day_low'] = df['prev_day_low'].fillna(method='ffill')
    
    # Sweep flags
    df['previousDayHighTaken'] = df['high'] > df['prev_day_high']
    df['previousDayLowTaken'] = df['low'] < df['prev_day_low']
    
    flagpdh = False
    flagpdl = False
    df['flagpdh'] = False
    df['flagpdl'] = False
    for i in range(len(df)):
        if df['previousDayHighTaken'].iloc[i]:
            flagpdh = True
            flagpdl = False
        elif df['previousDayLowTaken'].iloc[i]:
            flagpdl = True
            flagpdh = False
        else:
            flagpdl = False
            flagpdh = False
        df['flagpdh'].iloc[i] = flagpdh
        df['flagpdl'].iloc[i] = flagpdl
    
    # Bull/Bear G
    df['bullG'] = df['low'] > df['high'].shift(1)
    df['bearG'] = df['high'] < df['low'].shift(1)
    
    # ATR for FVG
    df['atr'] = atr_144 * fvgTH
    
    # FVG detection
    df['bullFvgUpper'] = np.nan
    df['bullFvgLower'] = np.nan
    df['bearFvgUpper'] = np.nan
    df['bearFvgLower'] = np.nan
    df['last'] = np.nan
    df['fvgActive'] = False
    df['fvgPct'] = 0.0
    
    fvgActive = False
    last = False
    bullFvgUpper = np.nan
    bullFvgLower = np.nan
    bearFvgUpper = np.nan
    bearFvgLower = np.nan
    
    for i in range(2, len(df)):
        bullG = df['bullG'].iloc[i]
        bullG1 = df['bullG'].iloc[i-1]
        bearG = df['bearG'].iloc[i]
        bearG1 = df['bearG'].iloc[i-1]
        atr_val = df['atr'].iloc[i]
        
        bull = False
        bear = False
        
        if not np.isnan(atr_val):
            bull = (df['low'].iloc[i] - df['high'].iloc[i-2]) > atr_val and df['low'].iloc[i] > df['high'].iloc[i-2] and df['close'].iloc[i-1] > df['high'].iloc[i-2] and not (bullG or bullG1)
            bear = (df['low'].iloc[i-2] - df['high'].iloc[i]) > atr_val and df['high'].iloc[i] < df['low'].iloc[i-2] and df['close'].iloc[i-1] < df['low'].iloc[i-2] and not (bearG or bearG1)
        
        if bull:
            bullFvgUpper = df['high'].iloc[i-2]
            bullFvgLower = df['low'].iloc[i]
            bearFvgUpper = np.nan
            bearFvgLower = np.nan
            fvgActive = True
            last = True
            df.at[df.index[i], 'fvgActive'] = True
            df.at[df.index[i], 'last'] = True
        elif bear:
            bearFvgUpper = df['high'].iloc[i]
            bearFvgLower = df['low'].iloc[i-2]
            bullFvgUpper = np.nan
            bullFvgLower = np.nan
            fvgActive = True
            last = False
            df.at[df.index[i], 'fvgActive'] = True
            df.at[df.index[i], 'last'] = False
        
        if bullG or bearG:
            fvgActive = False
        
        if fvgActive:
            cUB = bullFvgUpper if last else bearFvgUpper
            cLB = bullFvgLower if last else bearFvgLower
            tUB = bullFvgUpper if last else bearFvgUpper
            bMB = bullFvgLower if last else bearFvgLower
            if not np.isnan(cUB) and not np.isnan(cLB):
                pct = (tUB - bMB) / (cUB - cLB)
                df.at[df.index[i], 'fvgPct'] = pct
        
        df.at[df.index[i], 'bullFvgUpper'] = bullFvgUpper
        df.at[df.index[i], 'bullFvgLower'] = bullFvgLower
        df.at[df.index[i], 'bearFvgUpper'] = bearFvgUpper
        df.at[df.index[i], 'bearFvgLower'] = bearFvgLower
    
    # Midpoints
    df['bullMidpoint'] = np.nan
    df['bearMidpoint'] = np.nan
    
    for i in range(1, len(df)):
        bullFu = df['bullFvgUpper'].iloc[i]
        bullL = df['bullFvgLower'].iloc[i]
        bearU = df['bearFvgUpper'].iloc[i]
        bearL = df['bearFvgLower'].iloc[i]
        
        if not np.isnan(bullFu) and not np.isnan(bullL):
            df.at[df.index[i], 'bullMidpoint'] = (bullFu + bullL) / 2
        if not np.isnan(bearU) and not np.isnan(bearL):
            df.at[df.index[i], 'bearMidpoint'] = (bearU + bearL) / 2
    
    # Trading window
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['total_minutes'] = df['hour'] * 60 + df['minute']
    morning_start = 7 * 60 + 45
    morning_end = 9 * 60 + 45
    afternoon_start = 14 * 60 + 45
    afternoon_end = 16 * 60 + 45
    df['isWithinTimeWindow'] = ((df['total_minutes'] >= morning_start) & (df['total_minutes'] < morning_end)) | ((df['total_minutes'] >= afternoon_start) & (df['total_minutes'] < afternoon_end))
    
    # Entry conditions
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if df['flagpdh'].iloc[i] and df['isWithinTimeWindow'].iloc[i] and df['fvgActive'].iloc[i] and df['fvgPct'].iloc[i] > 0.01 and df['fvgPct'].iloc[i] <= 1:
            last_val = df['last'].iloc[i]
            if last_val == False:
                bear_mid = df['bearMidpoint'].iloc[i]
                if not np.isnan(bear_mid) and df['high'].iloc[i] < bear_mid:
                    ts = int(df['time'].iloc[i])
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': ts,
                        'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                        'entry_price_guess': df['close'].iloc[i],
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': df['close'].iloc[i],
                        'raw_price_b': df['close'].iloc[i]
                    })
                    trade_num += 1
        
        if df['flagpdl'].iloc[i] and df['isWithinTimeWindow'].iloc[i] and df['fvgActive'].iloc[i] and df['fvgPct'].iloc[i] > 0.01 and df['fvgPct'].iloc[i] <= 1:
            last_val = df['last'].iloc[i]
            if last_val == True:
                bull_mid = df['bullMidpoint'].iloc[i]
                if not np.isnan(bull_mid) and df['low'].iloc[i] > bull_mid:
                    ts = int(df['time'].iloc[i])
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': ts,
                        'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                        'entry_price_guess': df['close'].iloc[i],
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': df['close'].iloc[i],
                        'raw_price_b': df['close'].iloc[i]
                    })
                    trade_num += 1
    
    return entries