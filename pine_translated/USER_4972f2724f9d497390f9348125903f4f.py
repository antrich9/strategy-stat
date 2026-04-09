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
    
    def wilder_atr(high, low, close, period=14):
        high = high.astype(float)
        low = low.astype(float)
        close = close.astype(float)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = pd.Series(np.nan, index=high.index)
        atr.iloc[period-1] = tr.iloc[:period].mean()
        for i in range(period, len(tr)):
            atr.iloc[i] = (atr.iloc[i-1] * (period - 1) + tr.iloc[i]) / period
        return atr
    
    atr14 = wilder_atr(df['high'], df['low'], df['close'], 14)
    atr20 = wilder_atr(df['high'], df['low'], df['close'], 20)
    
    vol_filt = (df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5)
    atr211 = atr20 / 1.5
    atr_filt = ((df['low'] - df['high'].shift(2) > atr211) | (df['low'].shift(2) - df['high'] > atr211))
    loc11 = df['close'].rolling(54).mean()
    loc211 = loc11 > loc11.shift(1)
    trend_filt_bull = loc211
    trend_filt_bear = ~loc211
    
    bull_fvg = (df['low'] > df['high'].shift(2)) & vol_filt & atr_filt & trend_filt_bull
    bear_fvg = (df['high'] < df['low'].shift(2)) & vol_filt & atr_filt & trend_filt_bear
    
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['datetime'].dt.date
    
    daily_agg = df.groupby('date').agg(
        daily_high=('high', 'max'),
        daily_low=('low', 'min')
    ).shift(1)
    daily_agg['date'] = daily_agg.index
    df = df.merge(daily_agg, on='date', how='left')
    
    df['is_new_day'] = df['date'] != df['date'].shift(1)
    first_bar_of_day = df['is_new_day'] & ~df['is_new_day'].shift(1).fillna(True)
    
    swing_high = pd.Series(np.nan, index=df.index)
    swing_low = pd.Series(np.nan, index=df.index)
    last_swing_type = "none"
    
    for i in range(4, len(df)):
        if pd.isna(df['daily_high'].iloc[i]) or pd.isna(df['daily_low'].iloc[i]):
            continue
        daily_h1 = df['high'].iloc[i-1]
        daily_h2 = df['high'].iloc[i-2]
        daily_h3 = df['high'].iloc[i-3]
        daily_h4 = df['high'].iloc[i-4]
        daily_l1 = df['low'].iloc[i-1]
        daily_l2 = df['low'].iloc[i-2]
        daily_l3 = df['low'].iloc[i-3]
        daily_l4 = df['low'].iloc[i-4]
        if daily_h1 < daily_h2 and daily_h3 < daily_h2 and daily_h4 < daily_h2:
            swing_high.iloc[i] = daily_h2
            if daily_h2 < swing_low.iloc[i-1] if not pd.isna(swing_low.iloc[i-1]) else True:
                last_swing_type = "dailyLow"
        if daily_l1 > daily_l2 and daily_l3 > daily_l2 and daily_l4 > daily_l2:
            swing_low.iloc[i] = daily_l2
            if daily_l2 > swing_high.iloc[i-1] if not pd.isna(swing_high.iloc[i-1]) else True:
                last_swing_type = "dailyHigh"
    
    supertrend_multiplier = 3
    supertrend_period = 10
    supertrend_val = pd.Series(np.nan, index=df.index)
    supertrend_dir = pd.Series(0, index=df.index)
    upper_band = df['close'] + supertrend_multiplier * atr14
    lower_band = df['close'] - supertrend_multiplier * atr14
    prev_close = df['close'].shift(1)
    bull_cond = df['close'] > prev_close
    bear_cond = df['close'] < prev_close
    supertrend_dir.iloc[supertrend_period] = 1
    supertrend_val.iloc[supertrend_period] = lower_band.iloc[supertrend_period]
    for i in range(supertrend_period + 1, len(df)):
        if bull_cond.iloc[i]:
            supertrend_dir.iloc[i] = 1
        elif bear_cond.iloc[i]:
            supertrend_dir.iloc[i] = -1
        else:
            supertrend_dir.iloc[i] = supertrend_dir.iloc[i-1]
        if supertrend_dir.iloc[i] == 1:
            supertrend_val.iloc[i] = lower_band.iloc[i] if lower_band.iloc[i] > supertrend_val.iloc[i-1] or supertrend_dir.iloc[i-1] == -1 else supertrend_val.iloc[i-1]
        else:
            supertrend_val.iloc[i] = upper_band.iloc[i] if upper_band.iloc[i] < supertrend_val.iloc[i-1] or supertrend_dir.iloc[i-1] == 1 else supertrend_val.iloc[i-1]
    is_supertrend_bullish = supertrend_dir == 1
    is_supertrend_bearish = supertrend_dir == -1
    
    london_start_morning_hour, london_start_morning_min = 6, 45
    london_end_morning_hour, london_end_morning_min = 9, 45
    london_start_afternoon1_hour, london_start_afternoon1_min = 10, 45
    london_end_afternoon1_hour, london_end_afternoon1_min = 11, 45
    london_start_afternoon_hour, london_start_afternoon_min = 14, 45
    london_end_afternoon_hour, london_end_afternoon_min = 16, 45
    
    trades = []
    trade_num = 1
    
    for i in range(4, len(df)):
        ts = int(df['time'].iloc[i])
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour, minute = dt.hour, dt.minute
        in_morning = (hour > london_start_morning_hour or (hour == london_start_morning_hour and minute >= london_start_morning_min)) and \
                     (hour < london_end_morning_hour or (hour == london_end_morning_hour and minute < london_end_morning_min))
        in_afternoon1 = (hour > london_start_afternoon1_hour or (hour == london_start_afternoon1_hour and minute >= london_start_afternoon1_min)) and \
                        (hour < london_end_afternoon1_hour or (hour == london_end_afternoon1_hour and minute < london_end_afternoon1_min))
        in_afternoon = (hour > london_start_afternoon_hour or (hour == london_start_afternoon_hour and minute >= london_start_afternoon_min)) and \
                       (hour < london_end_afternoon_hour or (hour == london_end_afternoon_hour and minute < london_end_afternoon_min))
        in_trading_window = in_afternoon
        
        if in_trading_window:
            prev_day_high = df['daily_high'].iloc[i] if not pd.isna(df['daily_high'].iloc[i]) else np.inf
            prev_day_low = df['daily_low'].iloc[i] if not pd.isna(df['daily_low'].iloc[i]) else -np.inf
            swept_high = df['high'].iloc[i] > prev_day_high if prev_day_high != np.inf else False
            swept_low = df['low'].iloc[i] < prev_day_low if prev_day_low != -np.inf else False
            
            long_entry = bull_fvg.iloc[i] and last_swing_type == "dailyLow" and is_supertrend_bullish.iloc[i] and not swept_high
            short_entry = bear_fvg.iloc[i] and last_swing_type == "dailyHigh" and is_supertrend_bearish.iloc[i] and not swept_low
            
            if long_entry or short_entry:
                direction = 'long' if long_entry else 'short'
                entry_price = df['close'].iloc[i]
                entry_time = dt.isoformat()
                trades.append({
                    'trade_num': trade_num,
                    'direction': direction,
                    'entry_ts': ts,
                    'entry_time': entry_time,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
    
    return trades