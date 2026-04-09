import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    swing_len = 10
    sweep_wick = True
    choch_bars = 20
    ob_scan_bars = 15
    cooldown = 1
    
    results = []
    trade_num = 1
    last_sig_bar = -cooldown
    
    pivot_highs = np.full(len(df), np.nan)
    pivot_lows = np.full(len(df), np.nan)
    
    for i in range(swing_len, len(df) - swing_len):
        if df['high'].iloc[i] == df['high'].iloc[i-swing_len:i+swing_len+1].max():
            pivot_highs[i] = df['high'].iloc[i]
        if df['low'].iloc[i] == df['low'].iloc[i-swing_len:i+swing_len+1].min():
            pivot_lows[i] = df['low'].iloc[i]
    
    setups = []
    
    for i in range(len(df)):
        if not np.isnan(pivot_highs[i]):
            recent_high = pivot_highs[i]
        if not np.isnan(pivot_lows[i]):
            recent_low = pivot_lows[i]
        
        bsl_swept = False
        ssl_swept = False
        
        if not np.isnan(recent_high):
            if sweep_wick:
                if df['high'].iloc[i] > recent_high and df['close'].iloc[i] < recent_high:
                    bsl_swept = True
            else:
                if df['high'].iloc[i] > recent_high:
                    bsl_swept = True
        
        if not np.isnan(recent_low):
            if sweep_wick:
                if df['low'].iloc[i] < recent_low and df['close'].iloc[i] > recent_low:
                    ssl_swept = True
            else:
                if df['low'].iloc[i] < recent_low:
                    ssl_swept = True
        
        if bsl_swept:
            setups.insert(0, {
                'is_bear': True,
                'swept_level': recent_high,
                'sweep_bar': i,
                'choch_level': recent_low,
                'state': 'SWEEP',
                'state_bar': i
            })
        
        if ssl_swept:
            setups.insert(0, {
                'is_bear': False,
                'swept_level': recent_low,
                'sweep_bar': i,
                'choch_level': recent_high,
                'state': 'SWEEP',
                'state_bar': i
            })
        
        if len(setups) > 8:
            setups.pop()
        
        new_setups = []
        for setup in setups:
            if setup['state'] == 'SWEEP' and (i - setup['state_bar']) > choch_bars:
                setup['state'] = 'DONE'
            
            if setup['state'] == 'SWEEP':
                ob_top = np.nan
                ob_bot = np.nan
                
                if setup['is_bear']:
                    if df['close'].iloc[i] < setup['choch_level']:
                        for lb in range(1, ob_scan_bars + 1):
                            if i - lb >= 0:
                                if df['close'].iloc[i - lb] > df['open'].iloc[i - lb]:
                                    cand_high = df['high'].iloc[i - lb]
                                    cand_low = min(df['open'].iloc[i - lb], df['close'].iloc[i - lb])
                                    if cand_low <= setup['swept_level'] <= cand_high:
                                        ob_top = cand_high
                                        ob_bot = cand_low
                                        break
                else:
                    if df['close'].iloc[i] > setup['choch_level']:
                        for lb in range(1, ob_scan_bars + 1):
                            if i - lb >= 0:
                                if df['close'].iloc[i - lb] < df['open'].iloc[i - lb]:
                                    cand_high = max(df['open'].iloc[i - lb], df['close'].iloc[i - lb])
                                    cand_low = df['low'].iloc[i - lb]
                                    if cand_low <= setup['swept_level'] <= cand_high:
                                        ob_top = cand_high
                                        ob_bot = cand_low
                                        break
                
                if not np.isnan(ob_top):
                    setup['state'] = 'DONE'
                    if i - last_sig_bar >= cooldown:
                        last_sig_bar = i
                        entry_price = df['close'].iloc[i]
                        entry_ts = int(df['time'].iloc[i])
                        entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
                        
                        results.append({
                            'trade_num': trade_num,
                            'direction': 'short',
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
            
            if setup['state'] != 'DONE' or i - setup['sweep_bar'] < 100:
                new_setups.append(setup)
        
        setups = new_setups
    
    return results