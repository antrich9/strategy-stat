import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    entries = []
    trade_num = 0
    
    bar_ms = 240 * 60 * 1000
    barhtf_ms = 240 * 60 * 1000
    offset = int(barhtf_ms / bar_ms)
    if offset < 1:
        offset = 1
    
    threshold1 = 5.0
    
    body_top = np.where(df['close'].values >= df['open'].values, df['close'].values, df['open'].values)
    body_bottom = np.where(df['close'].values >= df['open'].values, df['open'].values, df['close'].values)
    
    bull_wick_size = np.zeros(len(df))
    bear_wick_size = np.zeros(len(df))
    for i in range(len(df)):
        if body_top[i] > body_bottom[i]:
            bull_wick_size[i] = ((body_bottom[i] - df['low'].iloc[i]) / body_bottom[i]) * 100
            bear_wick_size[i] = ((df['high'].iloc[i] - body_top[i]) / body_top[i]) * 100
        else:
            bull_wick_size[i] = ((df['open'].iloc[i] - df['low'].iloc[i]) / df['open'].iloc[i]) * 100
            bear_wick_size[i] = ((df['high'].iloc[i] - df['open'].iloc[i]) / df['open'].iloc[i]) * 100
    
    bull_fvg_zones = []
    bear_fvg_zones = []
    bull_wick_zones = []
    bear_wick_zones = []
    
    for i in range(1, len(df)):
        ts = int(df['time'].iloc[i])
        entry_price = float(df['close'].iloc[i])
        
        if pd.isna(entry_price) or pd.isna(df['high'].iloc[i]) or pd.isna(df['low'].iloc[i]):
            continue
        
        if i < offset:
            continue
        
        bull_fvg = df['low'].iloc[i] > df['high'].iloc[i - offset]
        bear_fvg = df['high'].iloc[i] < df['low'].iloc[i - offset]
        
        if bull_fvg:
            zone_high = float(df['high'].iloc[i - offset])
            zone_low = float(df['low'].iloc[i])
            bull_fvg_zones.append({'zone_high': zone_high, 'zone_low': zone_low, 'bar_time': ts})
        
        if bear_fvg:
            zone_high = float(df['high'].iloc[i])
            zone_low = float(df['low'].iloc[i - offset])
            bear_fvg_zones.append({'zone_high': zone_high, 'zone_low': zone_low, 'bar_time': ts})
        
        if not pd.isna(bull_wick_size[i]) and bull_wick_size[i] >= threshold1:
            zone_high = float(body_top[i])
            zone_low = float(df['low'].iloc[i])
            bull_wick_zones.append({'zone_high': zone_high, 'zone_low': zone_low, 'bar_time': ts})
        
        if not pd.isna(bear_wick_size[i]) and bear_wick_size[i] >= threshold1:
            zone_high = float(df['high'].iloc[i])
            zone_low = float(body_bottom[i])
            bear_wick_zones.append({'zone_high': zone_high, 'zone_low': zone_low, 'bar_time': ts})
        
        for zone in list(bull_fvg_zones):
            if not pd.isna(zone['zone_low']) and entry_price < zone['zone_low']:
                trade_num += 1
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                bull_fvg_zones.remove(zone)
        
        for zone in list(bear_fvg_zones):
            if not pd.isna(zone['zone_high']) and entry_price > zone['zone_high']:
                trade_num += 1
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                bear_fvg_zones.remove(zone)
        
        for zone in list(bull_wick_zones):
            if not pd.isna(zone['zone_low']) and entry_price < zone['zone_low']:
                trade_num += 1
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                bull_wick_zones.remove(zone)
        
        for zone in list(bear_wick_zones):
            if not pd.isna(zone['zone_high']) and entry_price > zone['zone_high']:
                trade_num += 1
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                bear_wick_zones.remove(zone)
    
    return entries