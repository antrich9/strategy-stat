import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    swing_length = 5
    max_pda_age = 100
    wick_threshold = 0.3
    
    entries = []
    trade_num = 1
    n = len(df)
    
    if n < swing_length * 2 + 3:
        return entries
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_arr = df['open'].values
    ts_arr = df['time'].values
    
    def is_bullish_rejection(idx):
        if idx < 0 or idx >= n:
            return False
        candle_range = high[idx] - low[idx]
        if candle_range == 0:
            return False
        lower_wick = min(close[idx], open_arr[idx]) - low[idx]
        return (lower_wick / candle_range) >= wick_threshold
    
    def is_bearish_rejection(idx):
        if idx < 0 or idx >= n:
            return False
        candle_range = high[idx] - low[idx]
        if candle_range == 0:
            return False
        upper_wick = high[idx] - max(close[idx], open_arr[idx])
        return (upper_wick / candle_range) >= wick_threshold
    
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    for i in range(swing_length, n - swing_length):
        if high[i] == high[i-swing_length:i+swing_length+1].max():
            swing_high[i] = True
        if low[i] == low[i-swing_length:i+swing_length+1].min():
            swing_low[i] = True
    
    bullish_fvg = np.zeros(n, dtype=bool)
    bearish_fvg = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if low[i] > high[i-2]:
            bullish_fvg[i] = True
        if high[i] < low[i-2]:
            bearish_fvg[i] = True
    
    pdas = []
    
    for i in range(n):
        ts = int(ts_arr[i])
        
        if bullish_fvg[i]:
            pda = {
                'pda_type': 'BULL_FVG', 'bottom': high[i-2], 'top': low[i],
                'mid_level': (low[i] + high[i-2]) / 2, 'bar_created': i - 2,
                'is_premium': False, 'state': 'PENDING',
                'first_candle_bar': None, 'first_candle_high': np.nan, 'first_candle_low': np.nan,
                'rejection_level': np.nan, 'rejection_bar': None, 'id': len(pdas)
            }
            pdas.append(pda)
        
        if bearish_fvg[i]:
            pda = {
                'pda_type': 'BEAR_FVG', 'bottom': low[i-2], 'top': high[i],
                'mid_level': (low[i-2] + high[i]) / 2, 'bar_created': i - 2,
                'is_premium': True, 'state': 'PENDING',
                'first_candle_bar': None, 'first_candle_high': np.nan, 'first_candle_low': np.nan,
                'rejection_level': np.nan, 'rejection_bar': None, 'id': len(pdas)
            }
            pdas.append(pda)
        
        if swing_high[i] and i >= swing_length:
            pda = {
                'pda_type': 'SWING_HIGH', 'bottom': high[i-swing_length] * 0.9995,
                'top': high[i-swing_length] * 1.0005, 'mid_level': high[i-swing_length],
                'bar_created': i - swing_length, 'is_premium': True, 'state': 'PENDING',
                'first_candle_bar': None, 'first_candle_high': np.nan, 'first_candle_low': np.nan,
                'rejection_level': np.nan, 'rejection_bar': None, 'id': len(pdas)
            }
            pdas.append(pda)
        
        if swing_low[i] and i >= swing_length:
            pda = {
                'pda_type': 'SWING_LOW', 'bottom': low[i-swing_length] * 0.9995,
                'top': low[i-swing_length] * 1.0005, 'mid_level': low[i-swing_length],
                'bar_created': i - swing_length, 'is_premium': False, 'state': 'PENDING',
                'first_candle_bar': None, 'first_candle_high': np.nan, 'first_candle_low': np.nan,
                'rejection_level': np.nan, 'rejection_bar': None, 'id': len(pdas)
            }
            pdas.append(pda)
        
        pdas_to_remove = []
        for pda in pdas:
            age = i - pda['bar_created']
            if age > max_pda_age:
                pdas_to_remove.append(pda['id'])
                continue
            if pda['state'] == 'DISRESPECTED':
                continue
            
            bottom = pda['bottom']
            top = pda['top']
            
            if top >= low[i] and bottom <= high[i]:
                if pda['state'] == 'PENDING':
                    if pda['is_premium'] and is_bearish_rejection(i):
                        pda['rejection_level'] = high[i]
                        pda['rejection_bar'] = i
                        pda['state'] = 'RESPECTING'
                    elif (not pda['is_premium']) and is_bullish_rejection(i):
                        pda['rejection_level'] = low[i]
                        pda['rejection_bar'] = i
                        pda['state'] = 'RESPECTING'
                    pda['first_candle_bar'] = i
                    pda['first_candle_high'] = high[i]
                    pda['first_candle_low'] = low[i]
                
                elif pda['state'] == 'CANDLE_1':
                    if i == pda['first_candle_bar'] + 1:
                        pda['state'] = 'CANDLE_2'
                        if pda['is_premium']:
                            if high[i] > pda['first_candle_high'] and is_bearish_rejection(i):
                                pda['rejection_level'] = high[i]
                                pda['rejection_bar'] = i
                                pda['state'] = 'RESPECTING'
                        else:
                            if low[i] < pda['first_candle_low'] and is_bullish_rejection(i):
                                pda['rejection_level'] = low[i]
                                pda['rejection_bar'] = i
                                pda['state'] = 'RESPECTING'
                    else:
                        pda['state'] = 'DISRESPECTED'
                
                elif pda['state'] == 'CANDLE_2':
                    if i > pda['first_candle_bar'] + 1:
                        if np.isnan(pda['rejection_level']):
                            pda['state'] = 'DISRESPECTED'
                
                elif pda['state'] == 'RESPECTING':
                    if not np.isnan(pda['rejection_level']):
                        prev_close = close[i-1] if i > 0 else close[0]
                        if pda['is_premium']:
                            if close[i] > pda['rejection_level'] and prev_close <= pda['rejection_level']:
                                pda['state'] = 'DISRESPECTED'
                                entry_price = pda['rejection_level']
                                entries.append({
                                    'trade_num': trade_num, 'direction': 'short',
                                    'entry_ts': ts, 'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                                    'entry_price_guess': entry_price, 'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
                                    'raw_price_a': entry_price, 'raw_price_b': entry_price
                                })
                                trade_num += 1
                        else:
                            if close[i] < pda['rejection_level'] and prev_close >= pda['rejection_level']:
                                pda['state'] = 'DISRESPECTED'
                                entry_price = pda['rejection_level']
                                entries.append({
                                    'trade_num': trade_num, 'direction': 'long',
                                    'entry_ts': ts, 'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                                    'entry_price_guess': entry_price, 'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
                                    'raw_price_a': entry_price, 'raw_price_b': entry_price
                                })
                                trade_num += 1
        
        pdas = [p for p in pdas if p['id'] not in pdas_to_remove]
    
    return entries