import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    data = df.copy().reset_index(drop=True)
    
    for col in ['time', 'open', 'high', 'low', 'close', 'volume']:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Volume SMA (9 period)
    vol_sma = data['volume'].shift(1).rolling(9).mean()
    data['volfilt'] = data['volume'].shift(1) > vol_sma * 1.5
    
    # ATR (Wilder's method, 20 period) / 1.5
    tr1 = data['high'] - data['low']
    tr2 = np.abs(data['high'] - data['close'].shift(1))
    tr3 = np.abs(data['low'] - data['close'].shift(1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.zeros(len(data))
    atr[0] = true_range.iloc[0]
    for i in range(1, len(data)):
        atr[i] = (atr[i-1] * 13 + true_range.iloc[i]) / 14
    data['atr'] = pd.Series(atr, index=data.index)
    data['atr_adj'] = data['atr'] / 1.5
    
    # Location/trend filter (SMA 54)
    loc = data['close'].rolling(54).mean()
    data['loc_up'] = loc > loc.shift(1)
    data['locfiltb'] = data['loc_up']
    data['locfilts'] = ~data['loc_up']
    
    # FVG conditions
    data['fvg_up'] = data['low'] > data['high'].shift(2)
    data['fvg_down'] = data['high'] < data['low'].shift(2)
    
    # Combined FVG with filters
    data['bfvg'] = data['fvg_up'] & data['volfilt'] & (
        (data['low'] - data['high'].shift(2) > data['atr_adj']) | 
        (data['low'].shift(2) - data['high'] > data['atr_adj'])
    ) & data['locfiltb']
    
    data['sfvg'] = data['fvg_down'] & data['volfilt'] & (
        (data['low'] - data['high'].shift(2) > data['atr_adj']) | 
        (data['low'].shift(2) - data['high'] > data['atr_adj'])
    ) & data['locfilts']
    
    # OB conditions (isObUp checks previous bar pattern)
    is_up = data['close'] > data['open']
    is_down = data['close'] < data['open']
    data['ob_up'] = is_down.shift(1) & is_up & (data['close'] > data['high'].shift(1))
    data['ob_down'] = is_up.shift(1) & is_down & (data['close'] < data['low'].shift(1))
    
    # Time window (London time: 7:45-9:45 and 14:45-16:45)
    data['dt'] = pd.to_datetime(data['time'], unit='ms', utc=True).dt.tz_convert('Europe/London')
    data['hour'] = data['dt'].dt.hour
    data['minute'] = data['dt'].dt.minute
    data['time_val'] = data['hour'] + data['minute'] / 60.0
    data['in_trading_window'] = (
        ((data['time_val'] >= 7.75) & (data['time_val'] < 9.75)) |
        ((data['time_val'] >= 14.75) & (data['time_val'] < 16.75))
    )
    
    # Entry conditions (bfvg with ob_up shifted by 1, sfvg with ob_down shifted by 1)
    data['long_entry'] = data['bfvg'] & data['ob_up'].shift(1) & data['in_trading_window']
    data['short_entry'] = data['sfvg'] & data['ob_down'].shift(1) & data['in_trading_window']
    
    # Generate entry list
    entries = []
    trade_num = 1
    
    for i in range(1, len(data)):
        if pd.isna(data['atr'].iloc[i]) or pd.isna(loc.iloc[i]):
            continue
        
        entry_price = float(data['close'].iloc[i])
        ts = int(data['time'].iloc[i])
        
        if data['long_entry'].iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif data['short_entry'].iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries