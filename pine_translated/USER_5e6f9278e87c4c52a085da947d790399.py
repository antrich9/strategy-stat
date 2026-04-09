import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.

    Returns list of dicts:
    [{'trade_num': int, 'direction': 'long' or 'short',
      'entry_ts': int, 'entry_time': str,
      'entry_price_guess': float,
      'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
      'raw_price_a': float, 'raw_price_b': float}]
    """
    pp = 5
    atr_len = 55
    atr_len_sl = 14
    atr_mult = 1.5
    
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/length, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    def wilder_atr(high, low, close, length):
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/length, adjust=False).mean()
        return atr
    
    atr = wilder_atr(df['high'], df['low'], df['close'], atr_len)
    atr_sl = wilder_atr(df['high'], df['low'], df['close'], atr_len_sl)
    
    def find_pivots(high, low, length):
        swing_high = pd.Series(False, index=high.index)
        swing_low = pd.Series(False, index=low.index)
        for i in range(length, len(high) - length):
            if high.iloc[i] >= high.iloc[i-length:i].max() and high.iloc[i] >= high.iloc[i+1:i+length+1].max():
                swing_high.iloc[i] = True
            if low.iloc[i] <= low.iloc[i-length:i].min() and low.iloc[i] <= low.iloc[i+1:i+length+1].min():
                swing_low.iloc[i] = True
        return swing_high, swing_low
    
    swing_high, swing_low = find_pivots(df['high'], df['low'], pp)
    
    zigzag_type = []
    zigzag_value = []
    zigzag_index = []
    zigzag_type_adv = []
    zigzag_value_adv = []
    zigzag_index_adv = []
    
    for i in range(len(df)):
        if swing_high.iloc[i] and swing_low.iloc[i]:
            if len(zigzag_type) == 0:
                zigzag_type.append('H')
                zigzag_value.append(df['high'].iloc[i])
                zigzag_index.append(i)
            elif zigzag_type[-1] in ['L', 'LL']:
                if df['low'].iloc[i] < zigzag_value[-1]:
                    zigzag_value[-1] = df['low'].iloc[i]
                    zigzag_index[-1] = i
                    if len(zigzag_value) > 2:
                        zigzag_type[-1] = 'HL' if zigzag_value[-2] < df['low'].iloc[i] else 'LL'
                    else:
                        zigzag_type[-1] = 'L'
                else:
                    zigzag_type.append('H' if len(zigzag_value) <= 2 or zigzag_value[-2] >= df['high'].iloc[i] else 'HH')
                    zigzag_value.append(df['high'].iloc[i])
                    zigzag_index.append(i)
            elif zigzag_type[-1] in ['H', 'HH']:
                if df['high'].iloc[i] > zigzag_value[-1]:
                    zigzag_value[-1] = df['high'].iloc[i]
                    zigzag_index[-1] = i
                    if len(zigzag_value) > 2:
                        zigzag_type[-1] = 'LH' if zigzag_value[-2] > df['high'].iloc[i] else 'HH'
                    else:
                        zigzag_type[-1] = 'H'
                else:
                    zigzag_type.append('L' if len(zigzag_value) <= 2 or zigzag_value[-2] <= df['low'].iloc[i] else 'LL')
                    zigzag_value.append(df['low'].iloc[i])
                    zigzag_index.append(i)
        elif swing_high.iloc[i]:
            if len(zigzag_type) == 0:
                zigzag_type.append('H')
                zigzag_value.append(df['high'].iloc[i])
                zigzag_index.append(i)
            elif zigzag_type[-1] in ['L', 'LL']:
                zigzag_type.append('H' if len(zigzag_value) <= 2 or zigzag_value[-2] >= df['high'].iloc[i] else 'HH')
                zigzag_value.append(df['high'].iloc[i])
                zigzag_index.append(i)
        elif swing_low.iloc[i]:
            if len(zigzag_type) == 0:
                zigzag_type.append('L')
                zigzag_value.append(df['low'].iloc[i])
                zigzag_index.append(i)
            elif zigzag_type[-1] in ['H', 'HH']:
                zigzag_type.append('L' if len(zigzag_value) <= 2 or zigzag_value[-2] <= df['low'].iloc[i] else 'LL')
                zigzag_value.append(df['low'].iloc[i])
                zigzag_index.append(i)
    
    major_bull_bos = False
    major_bear_bos = False
    major_bull_choch = False
    major_bear_choch = False
    minor_bull_bos = False
    minor_bear_bos = False
    minor_bull_choch = False
    minor_bear_choch = False
    
    major_highs = []
    major_lows = []
    major_high_idx = []
    major_low_idx = []
    minor_highs = []
    minor_lows = []
    minor_high_idx = []
    minor_low_idx = []
    
    for i in range(len(zigzag_type)):
        if zigzag_type[i] in ['H', 'HH']:
            major_highs.append(zigzag_value[i])
            major_high_idx.append(zigzag_index[i])
            if len(major_lows) >= 2 and len(major_highs) >= 2:
                if major_highs[-1] > major_highs[-2] and major_lows[-1] > major_lows[-2]:
                    if len(major_highs) > 2 and major_highs[-2] <= major_highs[-3]:
                        major_bear_choch = True
                    else:
                        major_bull_bos = True
        elif zigzag_type[i] in ['L', 'LL']:
            major_lows.append(zigzag_value[i])
            major_low_idx.append(zigzag_index[i])
            if len(major_highs) >= 2 and len(major_lows) >= 2:
                if major_lows[-1] < major_lows[-2] and major_highs[-1] < major_highs[-2]:
                    if len(major_lows) > 2 and major_lows[-2] >= major_lows[-3]:
                        major_bull_choch = True
                    else:
                        major_bear_bos = True
    
    db_triggered = False
    dt_triggered = False
    last_db_idx = -1
    last_dt_idx = -1
    
    entries = []
    trade_num = 1
    
    for i in range(10, len(df)):
        if pd.isna(atr.iloc[i]):
            continue
        
        curr_bull_bos = False
        curr_bear_bos = False
        curr_bull_choch = False
        curr_bear_choch = False
        
        for j in range(len(major_high_idx)):
            if major_high_idx[j] == i and j > 0:
                if len(major_highs) > j + 1 and len(major_lows) >= j:
                    if major_highs[j] > major_highs[j-1] and major_lows[j] > major_lows[j-1]:
                        curr_bull_bos = True
        for j in range(len(major_low_idx)):
            if major_low_idx[j] == i and j > 0:
                if len(major_lows) > j + 1 and len(major_highs) >= j:
                    if major_lows[j] < major_lows[j-1] and major_highs[j] < major_highs[j-1]:
                        curr_bear_bos = True
        
        for j in range(3, len(zigzag_type)):
            if zigzag_index[j] == i:
                if zigzag_type[j] in ['HH'] and zigzag_type[j-1] in ['LH']:
                    curr_bull_choch = True
                if zigzag_type[j] in ['LL'] and zigzag_type[j-1] in ['HL']:
                    curr_bear_choch = True
        
        if len(zigzag_type) >= 4:
            for j in range(len(zigzag_type)):
                if zigzag_index[j] == i:
                    if zigzag_type[j] == 'LL' and zigzag_type[j-1] == 'LH' and zigzag_type[j-2] == 'HH':
                        if df['close'].iloc[i] > df['close'].iloc[zigzag_index[j-3]] if j >= 3 and zigzag_index[j-3] < len(df) else False:
                            dt_triggered = True
                            last_dt_idx = i
                    if zigzag_type[j] == 'HH' and zigzag_type[j-1] == 'HL' and zigzag_type[j-2] == 'LL':
                        if df['close'].iloc[i] < df['close'].iloc[zigzag_index[j-3]] if j >= 3 and zigzag_index[j-3] < len(df) else False:
                            db_triggered = True
                            last_db_idx = i
        
        if dt_triggered:
            if i - last_dt_idx <= 10:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': df['close'].iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': df['close'].iloc[i],
                    'raw_price_b': df['close'].iloc[i]
                })
                trade_num += 1
                dt_triggered = False
        
        if db_triggered:
            if i - last_db_idx <= 10:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': df['close'].iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': df['close'].iloc[i],
                    'raw_price_b': df['close'].iloc[i]
                })
                trade_num += 1
                db_triggered = False
    
    return entries