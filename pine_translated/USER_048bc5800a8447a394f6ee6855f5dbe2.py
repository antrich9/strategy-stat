import pandas as pd
import numpy as np
from datetime import datetime, timezone

def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

def generate_entries(df: pd.DataFrame) -> list:
    if len(df) < 50:
        return []
    
    df = df.copy().sort_values('time').reset_index(drop=True)
    
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df_4h = df.set_index('datetime').resample('4h', origin='start').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().copy()
    
    volfilt = df_4h['volume'] > df_4h['volume'].ewm(span=9, adjust=False).mean() * 1.5
    atr = wilder_atr(df_4h['high'], df_4h['low'], df_4h['close'], 20) / 1.5
    atrfilt = (df_4h['low'] - df_4h['high'].shift(2) > atr) | (df_4h['low'].shift(2) - df_4h['high'] > atr)
    loc = df_4h['close'].rolling(54).mean()
    locfiltb = loc > loc.shift(1)
    locfilts = loc < loc.shift(1)
    
    bull_fvg1 = (df_4h['low'] > df_4h['high'].shift(2)) & (df_4h['close'].shift(1) > df_4h['high'].shift(2)) & volfilt & atrfilt & locfiltb
    bear_fvg1 = (df_4h['high'] < df_4h['low'].shift(2)) & (df_4h['close'].shift(1) < df_4h['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    bull_fvg_shifted = bull_fvg1.shift(1).fillna(False)
    bear_fvg_shifted = bear_fvg1.shift(1).fillna(False)
    
    bull_sharp_turn = bull_fvg1 & bear_fvg_shifted
    bear_sharp_turn = bear_fvg1 & bull_fvg_shifted
    
    df_4h['bull_sharp_turn'] = bull_sharp_turn
    df_4h['bear_sharp_turn'] = bear_sharp_turn
    df_4h['bull_fvg'] = bull_fvg1
    df_4h['bear_fvg'] = bear_fvg1
    
    last_fvg_state = 0
    last_fvg_arr = []
    for idx in df_4h.index:
        curr = df_4h.loc[idx]
        if curr['bear_fvg']:
            last_fvg_state = -1
        elif curr['bull_fvg']:
            last_fvg_state = 1
        last_fvg_arr.append(last_fvg_state)
    df_4h['last_fvg'] = last_fvg_arr
    df_4h['last_fvg_prev'] = df_4h['last_fvg'].shift(1).fillna(0)
    
    df_4h['bull_sharp_turn_v2'] = df_4h['bull_fvg'] & (df_4h['last_fvg_prev'] == -1)
    df_4h['bear_sharp_turn_v2'] = df_4h['bear_fvg'] & (df_4h['last_fvg_prev'] == 1)
    
    mask = df_4h['bull_sharp_turn_v2'] | df_4h['bear_sharp_turn_v2']
    signal_4h = df_4h[mask]
    
    entries = []
    trade_num = 1
    
    for ts in signal_4h.index:
        direction = 'long' if signal_4h.loc[ts, 'bull_sharp_turn_v2'] else 'short'
        entry_ts = int(ts.timestamp())
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        entry_price = signal_4h.loc[ts, 'close']
        
        entries.append({
            'trade_num': trade_num,
            'direction': direction,
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
    
    return entries