import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['datetime'].dt.date
    
    df['is_new_day'] = df['date'] != df['date'].shift(1)
    
    daily_high = df.groupby('date')['high'].cummax()
    daily_low = df.groupby('date')['low'].cummin()
    
    df['prev_day_high'] = daily_high.shift(1).where(df['is_new_day'].shift(1))
    df['prev_day_low'] = daily_low.shift(1).where(df['is_new_day'].shift(1))
    
    df['prev_day_high'] = df.groupby('date')['prev_day_high'].ffill()
    df['prev_day_low'] = df.groupby('date')['prev_day_low'].ffill()
    
    body_size_req_c3 = np.abs(df['close'] - df['open']) >= (np.abs(df['high'] - df['low']) * 0.7)
    body_size_req_c1 = np.abs(df['close'].shift(2) - df['open'].shift(2)) >= (np.abs(df['high'].shift(2) - df['low'].shift(2)) * 0.7)
    
    bullish_cond = ((df['low'].shift(2) < df['prev_day_low']) | (df['low'].shift(1) < df['prev_day_low'])) & \
                   (df['low'].shift(1) < df['low'].shift(2)) & \
                   (df['high'].shift(1) < df['high'].shift(2)) & \
                   (df['low'] > df['low'].shift(1)) & \
                   (df['close'] > df['high'].shift(1))
    
    valid_bullish = bullish_cond & body_size_req_c3 & body_size_req_c1
    bullish_bars = valid_bullish
    
    bearish_cond = ((df['high'].shift(2) > df['prev_day_high']) | (df['high'].shift(1) > df['prev_day_high'])) & \
                   (df['high'].shift(1) > df['high'].shift(2)) & \
                   (df['low'].shift(1) > df['low'].shift(2)) & \
                   (df['high'] < df['high'].shift(1)) & \
                   (df['close'] < df['low'].shift(1))
    
    valid_bearish = bearish_cond & body_size_req_c3 & body_size_req_c1
    bearish_bars = valid_bearish
    
    df['session_hour'] = df['datetime'].dt.tz_convert('Etc/GMT+5').dt.hour
    in_session = (df['session_hour'] >= 8) & (df['session_hour'] <= 11)
    
    close_lag = df['close'].shift(1)
    pdh_broken_bu = (close_lag <= df['prev_day_high']) & (df['close'] > df['prev_day_high'])
    pdl_broken_bu = (close_lag >= df['prev_day_low']) & (df['close'] < df['prev_day_low'])
    
    pdh_broken_in_session = df.groupby('date')['pdh_broken_bu'].transform(lambda x: x.cumsum()) > 0
    pdl_broken_in_session = df.groupby('date')['pdl_broken_bu'].transform(lambda x: x.cumsum()) > 0
    
    long_entry = in_session & pdl_broken_in_session & bullish_bars
    short_entry = in_session & pdh_broken_in_session & bearish_bars
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(df['prev_day_high'].iloc[i]) or pd.isna(df['prev_day_low'].iloc[i]):
            continue
        
        if long_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        
        if short_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries