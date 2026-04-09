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
    
    results = []
    trade_num = 1
    
    # Convert time to datetime for time-based filtering
    df = df.copy()
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Previous day high/low (shift by 1 for "yesterday")
    df['prevDayHigh'] = df['high'].shift(1)
    df['prevDayLow'] = df['low'].shift(1)
    
    # Flags for PDH/PDL swept
    df['previousDayHighTaken'] = df['high'] > df['prevDayHigh']
    df['previousDayLowTaken'] = df['low'] < df['prevDayLow']
    
    # Initialize flag columns
    df['flagpdh'] = False
    df['flagpdl'] = False
    
    # Track state for flags
    flagpdh = False
    flagpdl = False
    
    for i in range(1, len(df)):
        if df['previousDayHighTaken'].iloc[i]:
            flagpdh = True
        elif df['previousDayLowTaken'].iloc[i]:
            flagpdl = True
        else:
            flagpdl = False
            flagpdh = False
        df.loc[df.index[i], 'flagpdh'] = flagpdh
        df.loc[df.index[i], 'flagpdl'] = flagpdl
    
    # Engulfing patterns
    df['bullG'] = df['low'] > df['high'].shift(1)  # Bullish engulfing
    df['bearG'] = df['high'] < df['low'].shift(1)  # Bearish engulfing
    
    # Wilder's ATR(144)
    def wilder_atr(high, low, close, length):
        tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
        tr.iloc[0] = high.iloc[0] - low.iloc[0]
        atr = pd.Series(index=tr.index, dtype=float)
        atr.iloc[length-1] = tr.iloc[:length].mean()
        multiplier = 1.0 / length
        for i in range(length, len(tr)):
            atr.iloc[i] = atr.iloc[i-1] * (1 - multiplier) + tr.iloc[i] * multiplier
        return atr
    
    atr_144 = wilder_atr(df['high'], df['low'], df['close'], 144)
    fvgTH = 0.5
    atr_threshold = atr_144 * fvgTH
    
    # FVG detection - bullish: (low - high[2]) > atr AND low > high[2] AND close[1] > high[2] AND NOT bullG AND NOT bullG[1]
    df['bullFvgCond'] = (
        (df['low'] - df['high'].shift(2)) > atr_threshold
    ) & (
        df['low'] > df['high'].shift(2)
    ) & (
        df['close'].shift(1) > df['high'].shift(2)
    ) & (
        ~df['bullG']
    ) & (
        ~df['bullG'].shift(1).fillna(False)
    )
    
    # FVG detection - bearish: (low[2] - high) > atr AND high < low[2] AND close[1] < low[2] AND NOT bearG AND NOT bearG[1]
    df['bearFvgCond'] = (
        (df['low'].shift(2) - df['high']) > atr_threshold
    ) & (
        df['high'] < df['low'].shift(2)
    ) & (
        df['close'].shift(1) < df['low'].shift(2)
    ) & (
        ~df['bearG']
    ) & (
        ~df['bearG'].shift(1).fillna(False)
    )
    
    # Track FVG state and midpoint
    df['bullFvgUpper'] = np.nan
    df['bullFvgLower'] = np.nan
    df['bearFvgUpper'] = np.nan
    df['bearFvgLower'] = np.nan
    df['bullMidpoint'] = np.nan
    df['bearMidpoint'] = np.nan
    df['fvgActive'] = False
    df['lastBullish'] = False  # Tracks direction of most recent FVG
    
    fvg_active = False
    last_bullish = False
    
    for i in range(2, len(df)):
        # FVG size check (simplified - check if we have recent FVG)
        has_recent_fvg = df['bullFvgCond'].iloc[i-2:i+1].any() or df['bearFvgCond'].iloc[i-2:i+1].any()
        
        if df['bullFvgCond'].iloc[i] and not fvg_active:
            df.loc[df.index[i], 'bullFvgUpper'] = df['high'].iloc[i-2]
            df.loc[df.index[i], 'bullFvgLower'] = df['low'].iloc[i]
            df.loc[df.index[i], 'bullMidpoint'] = (df['high'].iloc[i-2] + df['low'].iloc[i]) / 2
            fvg_active = True
            last_bullish = True
            df.loc[df.index[i], 'fvgActive'] = True
            df.loc[df.index[i], 'lastBullish'] = True
        elif df['bearFvgCond'].iloc[i] and not fvg_active:
            df.loc[df.index[i], 'bearFvgUpper'] = df['high'].iloc[i]
            df.loc[df.index[i], 'bearFvgLower'] = df['low'].iloc[i-2]
            df.loc[df.index[i], 'bearMidpoint'] = (df['high'].iloc[i] + df['low'].iloc[i-2]) / 2
            fvg_active = True
            last_bullish = False
            df.loc[df.index[i], 'fvgActive'] = True
            df.loc[df.index[i], 'lastBullish'] = False
        elif df['bullG'].iloc[i] or df['bearG'].iloc[i]:
            fvg_active = False
            df.loc[df.index[i], 'fvgActive'] = False
        else:
            df.loc[df.index[i], 'fvgActive'] = fvg_active
            df.loc[df.index[i], 'lastBullish'] = last_bullish
            # Propagate last midpoint if FVG still active
            if fvg_active:
                if last_bullish:
                    df.loc[df.index[i], 'bullMidpoint'] = df['bullMidpoint'].iloc[i-1]
                    df.loc[df.index[i], 'bullFvgUpper'] = df['bullFvgUpper'].iloc[i-1]
                    df.loc[df.index[i], 'bullFvgLower'] = df['bullFvgLower'].iloc[i-1]
                else:
                    df.loc[df.index[i], 'bearMidpoint'] = df['bearMidpoint'].iloc[i-1]
                    df.loc[df.index[i], 'bearFvgUpper'] = df['bearFvgUpper'].iloc[i-1]
                    df.loc[df.index[i], 'bearFvgLower'] = df['bearFvgLower'].iloc[i-1]
    
    # London trading windows
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    df['isMorningWindow'] = (df['hour'] == 7) & (df['minute'] >= 45) | (df['hour'] == 8) | (df['hour'] == 9) & (df['minute'] <= 45)
    df['isAfternoonWindow'] = (df['hour'] == 14) & (df['minute'] >= 45) | (df['hour'] == 15) | (df['hour'] == 16) & (df['minute'] <= 45)
    df['inTradingWindow'] = df['isMorningWindow'] | df['isAfternoonWindow']
    
    # Crossover/crossunder conditions
    df['bullCrossunder'] = (df['low'] < df['bullMidpoint']) & (df['low'].shift(1) >= df['bullMidpoint'].shift(1))
    df['bearCrossover'] = (df['high'] > df['bearMidpoint']) & (df['high'].shift(1) <= df['bearMidpoint'].shift(1))
    
    # Entry conditions
    df['longEntry'] = (
        df['flagpdh'] &
        df['inTradingWindow'] &
        df['bullCrossunder'] &
        df['fvgActive'] &
        df['lastBullish']
    )
    
    df['shortEntry'] = (
        (~df['lastBullish']) &
        df['fvgActive'] &
        df['bearCrossover']
    )
    
    # Generate entries
    for i in range(len(df)):
        if df['longEntry'].iloc[i] and not np.isnan(df['bullMidpoint'].iloc[i]):
            entry_price = df['bullMidpoint'].iloc[i]
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif df['shortEntry'].iloc[i] and not np.isnan(df['bearMidpoint'].iloc[i]):
            entry_price = df['bearMidpoint'].iloc[i]
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return results