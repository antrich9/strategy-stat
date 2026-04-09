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
    
    # Wilder RSI implementation
    def wilder_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Wilder ATR implementation
    def wilder_atr(high, low, close, period):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    # Input parameters (default values from Pine Script)
    PP = 5
    ATR_LENGTH_ZZ = 55
    ATR_LENGTH_SL = 14
    ATR_MULTIPLIER = 1.5
    
    # Calculate indicators
    high = df['high'].copy()
    low = df['low'].copy()
    close = df['close'].copy()
    
    atr_zz = wilder_atr(high, low, close, ATR_LENGTH_ZZ)
    atr_sl = wilder_atr(high, low, close, ATR_LENGTH_SL)
    
    # Pivot detection
    def get_pivot_highs_and_lows(high_series, low_series, period):
        pivots_h = (high_series == high_series.rolling(2*period+1, center=True).max()) & \
                   (high_series.shift(period) == high_series.iloc[period:].rolling(period, center=True).max().shift(-period))
        pivots_l = (low_series == low_series.rolling(2*period+1, center=True).min()) & \
                   (low_series.shift(period) == low_series.iloc[period:].rolling(period, center=True).min().shift(-period))
        
        pivots_h = pivots_h.fillna(False)
        pivots_l = pivots_l.fillna(False)
        
        pivot_high_idx = np.where(pivots_h)[0]
        pivot_low_idx = np.where(pivots_l)[0]
        
        return pivot_high_idx, pivot_low_idx
    
    pivot_high_idx, pivot_low_idx = get_pivot_highs_and_lows(high, low, PP)
    
    # Build ZigZag arrays (simplified version)
    zz_type = []
    zz_value = []
    zz_index = []
    
    for i in range(len(df)):
        if i in pivot_high_idx and i in pivot_low_idx:
            if len(zz_index) == 0:
                zz_type.append('H')
                zz_value.append(high.iloc[i])
                zz_index.append(i)
            else:
                last_type = zz_type[-1]
                last_val = zz_value[-1]
                if last_type in ['L', 'LL']:
                    if low.iloc[i] < last_val:
                        zz_type[-1] = 'L'
                        zz_value[-1] = low.iloc[i]
                        zz_index[-1] = i
                    else:
                        zz_type.append('H')
                        zz_value.append(high.iloc[i])
                        zz_index.append(i)
                else:
                    if high.iloc[i] > last_val:
                        zz_type.append('HH')
                        zz_value.append(high.iloc[i])
                        zz_index.append(i)
                    else:
                        zz_type.append('LH')
                        zz_value.append(high.iloc[i])
                        zz_index.append(i)
        elif i in pivot_high_idx:
            if len(zz_index) == 0:
                zz_type.append('H')
                zz_value.append(high.iloc[i])
                zz_index.append(i)
            else:
                last_type = zz_type[-1]
                last_val = zz_value[-1]
                if last_type in ['L', 'LL']:
                    if high.iloc[i] > low.iloc[zz_index[-1]]:
                        zz_type.append('H')
                        zz_value.append(high.iloc[i])
                        zz_index.append(i)
                    else:
                        zz_type.append('HH')
                        zz_value.append(high.iloc[i])
                        zz_index.append(i)
                else:
                    if high.iloc[i] > last_val:
                        if len(zz_index) >= 2 and zz_type[-2] in ['L', 'LL']:
                            zz_type.append('HH')
                        else:
                            zz_type.append('HH')
                        zz_value.append(high.iloc[i])
                        zz_index.append(i)
                    else:
                        zz_type.append('LH')
                        zz_value.append(high.iloc[i])
                        zz_index.append(i)
        elif i in pivot_low_idx:
            if len(zz_index) == 0:
                zz_type.append('L')
                zz_value.append(low.iloc[i])
                zz_index.append(i)
            else:
                last_type = zz_type[-1]
                last_val = zz_value[-1]
                if last_type in ['H', 'HH', 'LH']:
                    if low.iloc[i] < last_val:
                        zz_type.append('L')
                        zz_value.append(low.iloc[i])
                        zz_index.append(i)
                    else:
                        zz_type.append('LL')
                        zz_value.append(low.iloc[i])
                        zz_index.append(i)
                else:
                    if low.iloc[i] < last_val:
                        if len(zz_index) >= 2 and zz_type[-2] in ['H', 'HH', 'LH']:
                            zz_type.append('LL')
                        else:
                            zz_type.append('LL')
                        zz_value.append(low.iloc[i])
                        zz_index.append(i)
                    else:
                        zz_type.append('HL')
                        zz_value.append(low.iloc[i])
                        zz_index.append(i)
    
    # Create pivot-based signals
    pivot_high = pd.Series(np.nan, index=df.index)
    pivot_low = pd.Series(np.nan, index=df.index)
    
    for idx, val in zip(pivot_high_idx, high.iloc[pivot_high_idx]):
        pivot_high.iloc[idx] = val
    
    for idx, val in zip(pivot_low_idx, low.iloc[pivot_low_idx]):
        pivot_low.iloc[idx] = val
    
    pivot_high = pivot_high.fillna(method='ffill')
    pivot_low = pivot_low.fillna(method='ffill')
    
    # EMA calculations
    ema_fast = close.ewm(span=9, adjust=False).mean()
    ema_slow = close.ewm(span=21, adjust=False).mean()
    
    # Entry signals based on typical MSS/BoS/ZigZag strategy
    # Long entry: EMA crossover + price above recent low + ATR confirmation
    # Short entry: EMA crossunder + price below recent high + ATR confirmation
    
    entries = []
    trade_num = 1
    
    # Calculate conditions
    ema_crossover_long = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
    ema_crossunder_short = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
    
    # Price structure conditions
    price_above_pivot_low = close > pivot_low
    price_below_pivot_high = close < pivot_high
    
    # ATR filter
    atr_confirm_long = close > (close - atr_sl * ATR_MULTIPLIER)
    atr_confirm_short = close < (close + atr_sl * ATR_MULTIPLIER)
    
    # Combined entry conditions
    long_condition = ema_crossover_long & price_above_pivot_low & atr_confirm_long
    short_condition = ema_crossunder_short & price_below_pivot_high & atr_confirm_short
    
    # Build boolean series
    long_cond = long_condition & ~long_condition.shift(1).fillna(False)
    short_cond = short_condition & ~short_condition.shift(1).fillna(False)
    
    # Iterate and generate entries
    for i in range(len(df)):
        if pd.isna(atr_zz.iloc[i]) or pd.isna(atr_sl.iloc[i]):
            continue
            
        entry_price = close.iloc[i]
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        if long_cond.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if short_cond.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
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