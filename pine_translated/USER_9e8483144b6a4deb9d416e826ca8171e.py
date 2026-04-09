import pandas as pd
import numpy as np
from datetime import datetime, timezone

def wilder_rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    
    resampled_4h = df.resample('240min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    inp11 = False
    inp21 = False
    inp31 = False
    
    volfilt1 = True
    if inp11:
        vol_sma_4h = resampled_4h['volume'].rolling(9).mean()
        volfilt1 = resampled_4h['volume'].shift(1) > vol_sma_4h.shift(1) * 1.5
    
    atr_length1 = 20
    atr_4h = wilder_atr(resampled_4h['high'], resampled_4h['low'], resampled_4h['close'], atr_length1)
    atrfilt1 = True
    if inp21:
        atrfilt1 = ((resampled_4h['low'] - resampled_4h['high'].shift(2) > atr_4h / 1.5) | 
                   (resampled_4h['low'].shift(2) - resampled_4h['high'] > atr_4h / 1.5))
    
    loc1 = resampled_4h['close'].rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    locfiltb1 = True
    locfilts1 = True
    if inp31:
        locfiltb1 = loc21
        locfilts1 = ~loc21
    
    bfvg1 = (resampled_4h['low'] > resampled_4h['high'].shift(2)) & volfilt1 & atrfilt1 & locfiltb1
    sfvg1 = (resampled_4h['high'] < resampled_4h['low'].shift(2)) & volfilt1 & atrfilt1 & locfilts1
    
    entries_4h = []
    last_fvg = 0
    
    for i in range(1, len(resampled_4h)):
        current_bfvg = bfvg1.iloc[i]
        current_sfvg = sfvg1.iloc[i]
        
        if current_bfvg and last_fvg == -1:
            entries_4h.append({
                'direction': 'long',
                'timestamp': resampled_4h.index[i]
            })
            last_fvg = 1
        elif current_sfvg and last_fvg == 1:
            entries_4h.append({
                'direction': 'short',
                'timestamp': resampled_4h.index[i]
            })
            last_fvg = -1
        elif current_bfvg:
            last_fvg = 1
        elif current_sfvg:
            last_fvg = -1
    
    london_tz = timezone.utc
    london_start = datetime(2000, 1, 1, 7, 45, tzinfo=london_tz)
    london_end = datetime(2000, 1, 1, 9, 45, tzinfo=london_tz)
    
    def is_in_london_window(ts):
        london_time = ts.tz_convert(london_tz)
        hour_min = london_time.hour * 60 + london_time.minute
        start_min = 7 * 60 + 45
        end_min = 9 * 60 + 45
        return start_min <= hour_min < end_min
    
    filtered_entries = [e for e in entries_4h if is_in_london_window(e['timestamp'])]
    
    entries = []
    trade_num = 1
    
    for entry in filtered_entries:
        entry_ts_4h = entry['timestamp']
        direction = entry['direction']
        
        entry_start = entry_ts_4h
        entry_end = entry_ts_4h + pd.Timedelta(minutes=240)
        
        mask = (df.index >= entry_start) & (df.index < entry_end)
        bars_in_period = df[mask]
        
        if len(bars_in_period) > 0:
            first_bar = bars_in_period.iloc[0]
            entry_ts_int = int(first_bar['time'])
            entry_time_str = datetime.fromtimestamp(entry_ts_int, tz=timezone.utc).isoformat()
            entry_price = float(first_bar['close'])
        else:
            entry_ts_int = int(entry_ts_4h.timestamp())
            entry_time_str = entry_ts_4h.isoformat()
            entry_price = float(resampled_4h.loc[entry_ts_4h, 'close'])
        
        entries.append({
            'trade_num': trade_num,
            'direction': direction,
            'entry_ts': entry_ts_int,
            'entry_time': entry_time_str,
            'entry_price_guess': entry_price,
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': entry_price,
            'raw_price_b': entry_price
        })
        trade_num += 1
    
    return entries