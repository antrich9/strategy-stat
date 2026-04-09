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
    
    # Input settings from Pine Script
    inp11 = False  # Volume Filter
    inp21 = False  # ATR Filter
    inp31 = False  # Trend Filter
    
    lookback_bars = 12
    threshold = 0.0
    
    # Function to resample to 4H timeframe
    def resample_to_4h(data):
        period = 14400  # 4 hours in seconds
        data['period_start'] = (data['time'] // period) * period
        grouped = data.groupby('period_start').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        grouped = grouped.reset_index(drop=True)
        grouped['time'] = grouped['period_start'].values
        return grouped
    
    # Wilder RSI implementation
    def wilders_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Wilder ATR implementation
    def wilders_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/period, adjust=False).mean()
        return atr
    
    # Create 4H data
    df_4h = resample_to_4h(df.copy())
    
    # Detect new 4H candle
    df_4h['is_new_4h'] = True
    df_4h['prev_close'] = df_4h['close'].shift(1)
    
    # Volume Filter on 4H data
    sma_vol = df_4h['volume'].rolling(9).mean()
    volfilt1 = df_4h['volume'].shift(1) > sma_vol * 1.5
    
    # ATR Filter on 4H data
    atr_4h = wilders_atr(df_4h['high'], df_4h['low'], df_4h['close'], 20)
    atr_threshold = atr_4h / 1.5
    atr_gap_up = df_4h['low'] - df_4h['high'].shift(2)
    atr_gap_down = df_4h['low'].shift(2) - df_4h['high']
    atrfilt1 = (atr_gap_up > atr_threshold) | (atr_gap_down > atr_threshold)
    
    # Trend Filter on 4H data
    loc1 = df_4h['close'].rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    locfiltb1 = loc21
    locfilts1 = ~loc21
    
    # FVG detection on 4H data (bullish and bearish)
    bfvg1 = (df_4h['low'] > df_4h['high'].shift(2)) & volfilt1 & atrfilt1 & locfiltb1
    sfvg1 = (df_4h['high'] < df_4h['low'].shift(2)) & volfilt1 & atrfilt1 & locfilts1
    
    # Initialize FVG and sharp turn flags
    df_4h['lastFVG'] = 0
    df_4h['sharp_turn_long'] = False
    df_4h['sharp_turn_short'] = False
    
    # Detect sharp turns
    for i in range(1, len(df_4h)):
        if pd.isna(bfvg1.iloc[i]) or pd.isna(sfvg1.iloc[i]):
            continue
        if pd.isna(df_4h['lastFVG'].iloc[i-1]):
            prev_fvg = 0
        else:
            prev_fvg = int(df_4h['lastFVG'].iloc[i-1])
        
        if bfvg1.iloc[i] and prev_fvg == -1:
            df_4h.loc[df_4h.index[i], 'sharp_turn_long'] = True
            df_4h.loc[df_4h.index[i], 'lastFVG'] = 1
        elif sfvg1.iloc[i] and prev_fvg == 1:
            df_4h.loc[df_4h.index[i], 'sharp_turn_short'] = True
            df_4h.loc[df_4h.index[i], 'lastFVG'] = -1
        elif bfvg1.iloc[i]:
            df_4h.loc[df_4h.index[i], 'lastFVG'] = 1
        elif sfvg1.iloc[i]:
            df_4h.loc[df_4h.index[i], 'lastFVG'] = -1
    
    # Create time windows for London time
    def is_in_trading_window(timestamp):
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        total_minutes = hour * 60 + minute
        window1_start = 7 * 60 + 0
        window1_end = 11 * 60 + 45
        window2_start = 14 * 60 + 0
        window2_end = 14 * 60 + 45
        return (window1_start <= total_minutes < window1_end) or (window2_start <= total_minutes < window2_end)
    
    # FVG detection on chart timeframe (15m)
    df['volfilt'] = df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5
    df['atr_chart'] = wilders_atr(df['high'], df['low'], df['close'], 20)
    df['atr_gap_up_chart'] = df['low'] - df['high'].shift(2)
    df['atr_gap_down_chart'] = df['low'].shift(2) - df['high']
    df['atrfilt_chart'] = (df['atr_gap_up_chart'] > df['atr_chart'] / 1.5) | (df['atr_gap_down_chart'] > df['atr_chart'] / 1.5)
    loc_chart = df['close'].rolling(54).mean()
    df['loc2_chart'] = loc_chart > loc_chart.shift(1)
    df['locfiltb_chart'] = df['loc2_chart']
    df['locfilts_chart'] = ~df['loc2_chart']
    
    bfvg_chart = (df['low'] > df['high'].shift(2)) & df['volfilt'] & df['atrfilt_chart'] & df['locfiltb_chart']
    sfvg_chart = (df['high'] < df['low'].shift(2)) & df['volfilt'] & df['atrfilt_chart'] & df['locfilts_chart']
    
    # BPR conditions on chart timeframe
    bear_fvg1 = (df['high'] < df['low'].shift(2)) & (df['close'].shift(1) < df['low'].shift(2))
    bull_fvg1 = (df['low'] > df['high'].shift(2)) & (df['close'].shift(1) > df['high'].shift(2))
    
    def bars_since(series, target=True):
        result = pd.Series(-1, index=series.index)
        count = 0
        for i in range(len(series)):
            if series.iloc[i] == target:
                count = 0
            else:
                count += 1
            result.iloc[i] = count if series.iloc[i] != target else 0
        return result
    
    bull_since = bars_since(bfvg_chart, True)
    bull_cond_1 = bull_fvg1 & (bull_since <= lookback_bars)
    bull_since_vals = bull_since.where(bull_cond_1, -1)
    combined_low_bull = pd.Series(np.nan, index=df.index)
    combined_high_bull = pd.Series(np.nan, index=df.index)
    bull_result = pd.Series(False, index=df.index)
    
    for i in range(len(df)):
        if bull_cond_1.iloc[i] and bull_since_vals.iloc[i] >= 0:
            bs = int(bull_since_vals.iloc[i])
            if i - bs >= 0 and i - bs - 2 >= 0:
                combined_low_bull.iloc[i] = max(df['high'].iloc[i - bs], df['high'].iloc[i - bs - 2]) if not pd.isna(df['high'].iloc[i - bs]) and not pd.isna(df['high'].iloc[i - bs - 2]) else np.nan
                combined_high_bull.iloc[i] = min(df['low'].iloc[i - bs + 2], df['low'].iloc[i]) if i - bs + 2 <= i and not pd.isna(df['low'].iloc[i - bs + 2]) and not pd.isna(df['low'].iloc[i]) else np.nan
                if not pd.isna(combined_low_bull.iloc[i]) and not pd.isna(combined_high_bull.iloc[i]):
                    if combined_high_bull.iloc[i] - combined_low_bull.iloc[i] >= threshold:
                        bull_result.iloc[i] = True
    
    bear_since = bars_since(bull_fvg1, True)
    bear_cond_1 = bear_fvg1 & (bear_since <= lookback_bars)
    bear_since_vals = bear_since.where(bear_cond_1, -1)
    combined_low_bear = pd.Series(np.nan, index=df.index)
    combined_high_bear = pd.Series(np.nan, index=df.index)
    bear_result = pd.Series(False, index=df.index)
    
    for i in range(len(df)):
        if bear_cond_1.iloc[i] and bear_since_vals.iloc[i] >= 0:
            bs = int(bear_since_vals.iloc[i])
            if i - bs >= 0 and i - bs - 2 >= 0:
                combined_high_bear.iloc[i] = min(df['low'].iloc[i - bs], df['low'].iloc[i - bs - 2]) if not pd.isna(df['low'].iloc[i - bs]) and not pd.isna(df['low'].iloc[i - bs - 2]) else np.nan
                combined_low_bear.iloc[i] = max(df['high'].iloc[i - bs + 2], df['high'].iloc[i]) if i - bs + 2 <= i and not pd.isna(df['high'].iloc[i - bs + 2]) and not pd.isna(df['high'].iloc[i]) else np.nan
                if not pd.isna(combined_low_bear.iloc[i]) and not pd.isna(combined_high_bear.iloc[i]):
                    if combined_low_bear.iloc[i] - combined_high_bear.iloc[i] >= threshold:
                        bear_result.iloc[i] = True
    
    # Map 4H sharp turn signals to 15m bars
    df['sharp_turn_long'] = False
    df['sharp_turn_short'] = False
    
    for i in range(len(df_4h)):
        if df_4h['sharp_turn_long'].iloc[i] or df_4h['sharp_turn_short'].iloc[i]:
            ts_4h = df_4h['time'].iloc[i]
            next_ts_4h = ts_4h + 14400
            mask = (df['time'] >= ts_4h) & (df['time'] < next_ts_4h)
            if df_4h['sharp_turn_long'].iloc[i]:
                df.loc[mask, 'sharp_turn_long'] = True
            if df_4h['sharp_turn_short'].iloc[i]:
                df.loc[mask, 'sharp_turn_short'] = True
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        ts = df['time'].iloc[i]
        in_window = is_in_trading_window(ts)
        
        if not in_window:
            continue
        
        if df['sharp_turn_long'].iloc[i] or df['sharp_turn_short'].iloc[i]:
            if df['sharp_turn_long'].iloc[i]:
                direction = 'long'
                entry_price = df['close'].iloc[i]
                entries.append({
                    'trade_num': trade_num,
                    'direction': direction,
                    'entry_ts': int(ts),
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(entry_price),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(entry_price),
                    'raw_price_b': float(entry_price)
                })
                trade_num += 1
            elif df['sharp_turn_short'].iloc[i]:
                direction = 'short'
                entry_price = df['close'].iloc[i]
                entries.append({
                    'trade_num': trade_num,
                    'direction': direction,
                    'entry_ts': int(ts),
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(entry_price),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(entry_price),
                    'raw_price_b': float(entry_price)
                })
                trade_num += 1
    
    return entries