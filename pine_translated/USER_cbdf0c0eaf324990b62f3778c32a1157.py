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
    df = df.copy()
    df['ts'] = df['time']
    
    # Resample to 4H for FVG detection
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df_4h = df.set_index('time_dt').resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
    df_4h['high_4h'] = df_4h['high'].shift(1)
    df_4h['low_4h'] = df_4h['low'].shift(1)
    df_4h['close_4h'] = df_4h['close'].shift(1)
    df_4h['open_4h'] = df_4h['open'].shift(1)
    df_4h['high_4h_1'] = df_4h['high'].shift(2)
    df_4h['low_4h_1'] = df_4h['low'].shift(2)
    df_4h['high_4h_2'] = df_4h['high'].shift(3)
    df_4h['low_4h_2'] = df_4h['low'].shift(3)
    df_4h['close_4h_1'] = df_4h['close'].shift(2)
    
    # Resample to daily for swing detection
    df_daily = df.set_index('time_dt').resample('D').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
    df_daily['dailyHigh11'] = df_daily['high']
    df_daily['dailyLow11'] = df_daily['low']
    df_daily['dailyClose11'] = df_daily['close']
    df_daily['dailyOpen11'] = df_daily['open']
    df_daily['prevDayHigh11'] = df_daily['high'].shift(1)
    df_daily['prevDayLow11'] = df_daily['low'].shift(1)
    df_daily['dailyHigh21'] = df_daily['high'].shift(1)
    df_daily['dailyLow21'] = df_daily['low'].shift(1)
    df_daily['dailyHigh22'] = df_daily['high'].shift(2)
    df_daily['dailyLow22'] = df_daily['low'].shift(2)
    
    # Merge daily and 4H data back to main df
    df = df.merge(df_daily[['dailyHigh11', 'dailyLow11', 'dailyClose11', 'dailyOpen11', 'prevDayHigh11', 'prevDayLow11', 'dailyHigh21', 'dailyLow21', 'dailyHigh22', 'dailyLow22']], left_on='time_dt', right_index=True, how='left')
    df = df.merge(df_4h[['high_4h', 'low_4h', 'close_4h', 'open_4h', 'high_4h_1', 'low_4h_1', 'high_4h_2', 'low_4h_2', 'close_4h_1']], left_on='time_dt', right_index=True, how='left')
    
    # Swing detection
    df['is_swing_high11'] = (df['dailyHigh21'] < df['dailyHigh22']) & (df['dailyHigh11'].shift(3) < df['dailyHigh22']) & (df['dailyHigh11'].shift(4) < df['dailyHigh22'])
    df['is_swing_low11'] = (df['dailyLow21'] > df['dailyLow22']) & (df['dailyLow11'].shift(3) > df['dailyLow22']) & (df['dailyLow11'].shift(4) > df['dailyLow22'])
    
    # Track last swing high/low
    df['last_swing_high11'] = df['dailyHigh22'].where(df['is_swing_high11']).ffill()
    df['last_swing_low11'] = df['dailyLow22'].where(df['is_swing_low11']).ffill()
    
    # Detect new day
    df['newDay'] = df['time_dt'].dt.date != df['time_dt'].dt.date.shift(1)
    
    # Calculate previous day high/low manually
    df['pdHigh'] = np.where(df['newDay'], df['prevDayHigh11'].shift(1), np.nan)
    df['pdLow'] = np.where(df['newDay'], df['prevDayLow11'].shift(1), np.nan)
    
    # Forward fill pdHigh and pdLow
    df['pdHigh'] = df['pdHigh'].ffill()
    df['pdLow'] = df['pdLow'].ffill()
    
    # Sweep detection
    df['sweptHigh'] = (df['high'] > df['pdHigh'])
    df['sweptLow'] = (df['low'] < df['pdLow'])
    
    # Trading windows (Europe/London - UTC for simplicity, adjust as needed)
    df['hour'] = df['time_dt'].dt.hour
    df['minute'] = df['time_dt'].dt.minute
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    
    # Morning window: 06:45 - 08:55
    isWithinWindow1 = (df['time_minutes'] >= 405) & (df['time_minutes'] <= 535)
    # Afternoon window: 12:45 - 15:55
    isWithinWindow2 = (df['time_minutes'] >= 765) & (df['time_minutes'] <= 955)
    df['in_trading_window'] = isWithinWindow1 | isWithinWindow2
    
    # FVG conditions
    df['bfvg_condition'] = df['low_4h'] > df['high_4h_2']
    df['sfvg_condition'] = df['high_4h'] < df['low_4h_2']
    
    # Trend filter (loc1 > loc1[1])
    df['loc1'] = df['close_4h'].rolling(54).mean()
    df['loc21'] = df['loc1'] > df['loc1'].shift(1)
    df['locfiltb1'] = df['loc21']
    df['locfilts1'] = ~df['loc21']
    
    # ATR filter
    def wilder_atr(high, low, close, period=20):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    df['atr_4h'] = wilder_atr(df['high_4h'], df['low_4h'], df['close_4h'], 20) / 1.5
    df['atrfilt1'] = ((df['low_4h'] - df['high_4h_2'] > df['atr_4h']) | (df['low_4h_2'] - df['high_4h'] > df['atr_4h']))
    
    # Volume filter
    df['volfilt1'] = df['close_4h'].rolling(9).mean() * 1.5
    df['volfilt1'] = df['close_4h'].shift(1) > df['volfilt1']
    
    # Long entry conditions
    df['long_cond'] = df['bfvg_condition'] & df['sweptHigh'] & df['in_trading_window'] & df['locfiltb1']
    
    # Short entry conditions
    df['short_cond'] = df['sfvg_condition'] & df['sweptLow'] & df['in_trading_window'] & df['locfilts1']
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        if pd.isna(row['long_cond']) or pd.isna(row['short_cond']):
            continue
        
        if row['long_cond']:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(row['ts']),
                'entry_time': datetime.fromtimestamp(row['ts'], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(row['close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(row['close']),
                'raw_price_b': float(row['close'])
            })
            trade_num += 1
        elif row['short_cond']:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(row['ts']),
                'entry_time': datetime.fromtimestamp(row['ts'], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(row['close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(row['close']),
                'raw_price_b': float(row['close'])
            })
            trade_num += 1
    
    return entries