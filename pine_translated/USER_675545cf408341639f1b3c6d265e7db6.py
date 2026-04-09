import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    time = df['time']
    open_prices = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    MORNING_START = 8 * 3600
    MORNING_END = 9 * 3600 + 55 * 60
    AFTERNOON_START = 14 * 3600
    AFTERNOON_END = 16 * 3600 + 55 * 60
    
    def in_trading_window(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        in_morning = MORNING_START <= seconds < MORNING_END
        in_afternoon = AFTERNOON_START <= seconds < AFTERNOON_END
        return in_morning or in_afternoon
    
    signals = []
    trade_num = 1
    active_bull_zones = []
    active_bear_zones = []
    
    for i in range(len(df)):
        ts = int(time.iloc[i])
        
        if not in_trading_window(ts):
            continue
        
        if i >= 2:
            low_val_2 = low.iloc[i-2]
            high_val_2 = high.iloc[i-2]
            
            # Bullish FVG: low[i-2] >= high[i] (gap up)
            if low_val_2 >= high.iloc[i]:
                zone_top = low.iloc[i]
                zone_bottom = high_val_2
                active_bull_zones.append({
                    'top': zone_top,
                    'bottom': zone_bottom
                })
            
            # Bearish FVG: low[i] >= high[i-2] (gap down)
            if low.iloc[i] >= high_val_2:
                zone_top = low_val_2
                zone_bottom = high.iloc[i]
                active_bear_zones.append({
                    'top': zone_top,
                    'bottom': zone_bottom
                })
        
        for j in range(len(active_bull_zones)-1, -1, -1):
            zone = active_bull_zones[j]
            if low.iloc[i] < zone['top'] and low.iloc[i] >= zone['bottom']:
                signals.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close.iloc[i],
                    'raw_price_b': close.iloc[i]
                })
                trade_num += 1
                active_bull_zones.pop(j)
        
        for j in range(len(active_bear_zones)-1, -1, -1):
            zone = active_bear_zones[j]
            if high.iloc[i] > zone['bottom'] and high.iloc[i] <= zone['top']:
                signals.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close.iloc[i],
                    'raw_price_b': close.iloc[i]
                })
                trade_num += 1
                active_bear_zones.pop(j)
    
    return signals