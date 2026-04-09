import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    if len(df) == 0:
        return []
    
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_convert('Europe/London')
    
    df_4h = df.set_index('time').resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    df_4h = df_4h.rename(columns={'index': 'time'})
    df_4h['time'] = df_4h['time'].apply(lambda x: int(x.timestamp() * 1000))
    
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    close_4h = df_4h['close']
    volume_4h = df_4h['volume']
    
    def wilder_atr(high, low, close, length):
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr.iloc[0] = high.iloc[0] - low.iloc[0]
        atr = pd.Series(tr.ewm(alpha=1.0/length, adjust=False).mean(), index=tr.index)
        return atr
    
    def wilder_rsi(close, length):
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1.0/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    volfilt = volume_4h > pd.Series(volume_4h).rolling(9).mean() * 1.5
    atr_4h = wilder_atr(high_4h, low_4h, close_4h, 20) / 1.5
    atrfilt = ((low_4h - high_4h.shift(2) > atr_4h) | (low_4h.shift(2) - high_4h > atr_4h))
    loc = pd.Series(close_4h).rolling(54).mean()
    loc21 = loc > loc.shift(1)
    locfiltb = loc21
    locfilts = ~loc21
    bfvg = low_4h > high_4h.shift(2) & volfilt & atrfilt & locfiltb
    sfvg = high_4h < low_4h.shift(2) & volfilt & atrfilt & locfilts
    
    times = df_4h['time'].dt.hour * 60 + df_4h['time'].dt.minute
    in_window1 = (times >= 7 * 60 + 45) & (times < 11 * 60 + 45)
    in_window2 = (times >= 14 * 60) & (times < 14 * 60 + 45)
    in_trading_window = in_window1 | in_window2
    
    lastFVG = 0
    entries = []
    trade_num = 1
    
    for i in range(len(df_4h)):
        if pd.isna(bfvg.iloc[i]) and pd.isna(sfvg.iloc[i]):
            if not pd.isna(bfvg.iloc[i]):
                lastFVG = 1
            elif not pd.isna(sfvg.iloc[i]):
                lastFVG = -1
            continue
        
        current_bfvg = bfvg.iloc[i] if not pd.isna(bfvg.iloc[i]) else False
        current_sfvg = sfvg.iloc[i] if not pd.isna(sfvg.iloc[i]) else False
        
        if current_bfvg and lastFVG == -1:
            entry_ts = int(df_4h['time'].iloc[i].timestamp() * 1000)
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close_4h.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close_4h.iloc[i]),
                'raw_price_b': float(close_4h.iloc[i])
            })
            trade_num += 1
            lastFVG = 1
        elif current_sfvg and lastFVG == 1:
            entry_ts = int(df_4h['time'].iloc[i].timestamp() * 1000)
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close_4h.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close_4h.iloc[i]),
                'raw_price_b': float(close_4h.iloc[i])
            })
            trade_num += 1
            lastFVG = -1
        elif current_bfvg:
            lastFVG = 1
        elif current_sfvg:
            lastFVG = -1
    
    return entries