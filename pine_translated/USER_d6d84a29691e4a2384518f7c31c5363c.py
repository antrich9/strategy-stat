import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Previous Day High/Low
    daily_df = df.resample('D', on='datetime').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    daily_df['pdh'] = daily_df['high'].shift(1)
    daily_df['pdl'] = daily_df['low'].shift(1)
    daily_df['pdClose'] = daily_df['close'].shift(1)
    daily_df = daily_df.reset_index()
    daily_df['date'] = daily_df['datetime'].dt.date
    df['date'] = df['datetime'].dt.date
    df = df.merge(daily_df[['date', 'pdh', 'pdl', 'pdClose']], on='date', how='left')
    df['newDay'] = df['date'].diff() != pd.Timedelta(0)
    df['newDay'] = df['newDay'].fillna(False)
    
    # Bias tracking
    df['sweptLow'] = df['low'] < df['pdl']
    df['sweptHigh'] = df['high'] > df['pdh']
    df['brokeHigh'] = df['close'] > df['pdh']
    df['brokeLow'] = df['close'] < df['pdl']
    
    # Trading window (London hours)
    def get_london_hour(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        month = dt.month
        hour = dt.hour
        if 3 < month < 11:
            london_hour = hour - 1
            if london_hour < 0:
                london_hour += 24
        else:
            london_hour = hour
        return london_hour
    
    df['london_hour'] = df['time'].apply(get_london_hour)
    df['in_trading_window'] = (
        ((df['london_hour'] >= 7) & (df['london_hour'] < 12)) |
        ((df['london_hour'] >= 14) & (df['london_hour'] < 15))
    )
    
    # 4H data
    df['hour'] = df['datetime'].dt.hour
    df['4h_bucket'] = (df['hour'] // 4)
    df['date_hour'] = df['date'].astype(str) + '_' + df['4h_bucket'].astype(str)
    
    hf_df = df.groupby('date_hour').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'open': 'first',
        'volume': 'sum',
        'time': 'last'
    }).reset_index()
    hf_df = hf_df.sort_values('time').reset_index(drop=True)
    
    # Volume filter
    hf_df['vol_ma'] = hf_df['volume'].rolling(9).mean()
    hf_df['volfilt'] = hf_df['vol_ma'] * 1.5
    hf_df['volfilt_check'] = hf_df['volume'] > hf_df['volfilt']
    
    # ATR filter
    high_diff = hf_df['high'] - hf_df['low'].shift(2)
    low_diff = hf_df['low'].shift(2) - hf_df['high'].shift(1)
    tr1 = hf_df['high'] - hf_df['low']
    tr2 = (hf_df['high'] - hf_df['close'].shift(1)).abs()
    tr3 = (hf_df['low'] - hf_df['close'].shift(1)).abs()
    hf_df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    hf_df['atr'] = hf_df['tr'].ewm(alpha=1/20, adjust=False).mean()
    hf_df['atr_filt_val'] = hf_df['atr'] / 1.5
    hf_df['atrfilt'] = (high_diff > hf_df['atr_filt_val']) | (low_diff > hf_df['atr_filt_val'])
    
    # Trend filter
    hf_df['loc'] = hf_df['close'].rolling(54).mean()
    hf_df['loc_prev'] = hf_df['loc'].shift(1)
    hf_df['loc_up'] = hf_df['loc'] > hf_df['loc_prev']
    
    hf_df['bfvg'] = (hf_df['low'] > hf_df['high'].shift(2)) & hf_df['volfilt_check'] & hf_df['atrfilt'] & hf_df['loc_up']
    hf_df['sfvg'] = (hf_df['high'] < hf_df['low'].shift(2)) & hf_df['volfilt_check'] & hf_df['atrfilt'] & ~hf_df['loc_up']
    
    hf_df['new_4h'] = hf_df['date_hour'].shift(1) != hf_df['date_hour']
    hf_df['is_confirmed'] = True
    
    hf_df['prev_bfvg'] = hf_df['bfvg'].shift(1)
    hf_df['prev_sfvg'] = hf_df['sfvg'].shift(1)
    hf_df['prev_confirmed'] = hf_df['is_confirmed'].shift(1)
    hf_df['prev_new_4h'] = hf_df['new_4h'].shift(1)
    
    trades = []
    trade_num = 1
    lastFVG = 0
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        ts = row['time']
        date = row['date']
        date_hour = row['date_hour']
        
        if i == 1 or row['newDay']:
            lastFVG = 0
        
        hf_idx = hf_df[hf_df['date_hour'] == date_hour].index
        if len(hf_idx) == 0:
            continue
        
        idx = hf_idx[0]
        if idx < 2:
            continue
        
        hf_curr = hf_df.iloc[idx]
        hf_prev = hf_df.iloc[idx - 1]
        
        condition1 = hf_curr['is_confirmed'] and hf_curr['new_4h']
        condition2 = hf_curr['prev_confirmed'] and hf_curr['prev_new_4h']
        
        if not (condition1 or condition2):
            continue
        
        direction = None
        raw_price_a = 0.0
        raw_price_b = 0.0
        
        if hf_curr['bfvg'] and lastFVG == -1:
            direction = 'long'
            raw_price_a = row['close']
            raw_price_b = row['close']
        elif hf_curr['sfvg'] and lastFVG == 1:
            direction = 'short'
            raw_price_a = row['close']
            raw_price_b = row['close']
        elif hf_curr['bfvg']:
            lastFVG = 1
        elif hf_curr['sfvg']:
            lastFVG = -1
        
        if direction:
            trade = {
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': raw_price_a,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': raw_price_a,
                'raw_price_b': raw_price_b
            }
            trades.append(trade)
            trade_num += 1
            lastFVG = 1 if direction == 'long' else -1
    
    return trades