import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    inp1 = False  # Volume Filter
    inp2 = False  # ATR Filter
    inp3 = False  # Trend Filter
    lookback_bars = 12
    threshold = 0.0
    
    # Wilder's RSI
    def wilder_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Wilder's ATR
    def wilder_atr(high, low, close, period):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    # Calculate 4H data
    df_copy = df.copy()
    df_copy['ts_4h'] = pd.to_datetime(df_copy['time'], unit='s', utc=True)
    df_4h = df_copy.set_index('ts_4h').resample('240T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    close_4h = df_4h['close']
    volume_4h = df_4h['volume']
    
    # 4H Volume Filter
    volfilt_4h = volume_4h.shift(1) > volume_4h.rolling(9).mean() * 1.5
    if not inp1:
        volfilt_4h = pd.Series(True, index=volfilt_4h.index)
    
    # 4H ATR Filter
    atr_4h_raw = wilder_atr(high_4h, low_4h, close_4h, 20)
    atr_4h = atr_4h_raw / 1.5
    atrfilt_4h = ((low_4h - high_4h.shift(2) > atr_4h) | (low_4h.shift(2) - high_4h > atr_4h))
    if not inp2:
        atrfilt_4h = pd.Series(True, index=atrfilt_4h.index)
    
    # 4H Trend Filter
    loc = close_4h.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb_4h = loc2 if inp3 else pd.Series(True, index=loc2.index)
    locfilts_4h = ~loc2 if inp3 else pd.Series(True, index=loc2.index)
    
    # 4H Bullish/Bearish FVGs
    bfvg1 = (low_4h > high_4h.shift(2)) & volfilt_4h & atrfilt_4h & locfiltb_4h
    sfvg1 = (high_4h < low_4h.shift(2)) & volfilt_4h & atrfilt_4h & locfilts_4h
    
    # Sharp Turn detection
    lastFVG = pd.Series(0, index=df_4h.index)
    for i in range(1, len(df_4h)):
        prev_state = lastFVG.iloc[i-1]
        if bfvg1.iloc[i] and prev_state == -1:
            lastFVG.iloc[i] = 1
        elif sfvg1.iloc[i] and prev_state == 1:
            lastFVG.iloc[i] = -1
        elif bfvg1.iloc[i]:
            lastFVG.iloc[i] = 1
        elif sfvg1.iloc[i]:
            lastFVG.iloc[i] = -1
    
    # Sharp turn entries
    sharp_turn_long = bfvg1 & (lastFVG.shift(1) == -1)
    sharp_turn_short = sfvg1 & (lastFVG.shift(1) == 1)
    
    # BPR Logic (Bullish)
    bull_fvg1 = (low_4h > high_4h.shift(2)) & (close_4h.shift(1) < high_4h.shift(2))
    bull_since = pd.Series(0, index=df_4h.index)
    for i in range(len(df_4h)):
        if bull_fvg1.iloc[i]:
            bull_since.iloc[i] = 0
        elif i > 0:
            bull_since.iloc[i] = bull_since.iloc[i-1] + 1
    
    bull_cond_1 = bull_fvg1 & (bull_since <= lookback_bars)
    combined_low_bull = pd.Series(np.nan, index=df_4h.index)
    combined_high_bull = pd.Series(np.nan, index=df_4h.index)
    for i in range(len(df_4h)):
        if bull_cond_1.iloc[i]:
            bs = int(bull_since.iloc[i])
            combined_low_bull.iloc[i] = max(high_4h.iloc[i-bs] if i-bs >= 0 else 0, high_4h.iloc[i-2] if i-2 >= 0 else 0)
            combined_high_bull.iloc[i] = min(low_4h.iloc[i-bs+2] if i-bs+2 < len(low_4h) else float('inf'), low_4h.iloc[i])
    
    bull_result = bull_cond_1 & ((combined_high_bull - combined_low_bull) >= threshold)
    
    # Bearish BPR
    bear_fvg1_cond = (high_4h < low_4h.shift(2)) & (close_4h.shift(1) > low_4h.shift(2))
    
    # Chart timeframe FVG
    volfilt_chart = df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5
    if not inp1:
        volfilt_chart = pd.Series(True, index=volfilt_chart.index)
    
    atr_chart_raw = wilder_atr(df['high'], df['low'], df['close'], 20)
    atr_chart = atr_chart_raw / 1.5
    atrfilt_chart = ((df['low'] - df['high'].shift(2) > atr_chart) | (df['low'].shift(2) - df['high'] > atr_chart))
    if not inp2:
        atrfilt_chart = pd.Series(True, index=atrfilt_chart.index)
    
    loc_chart = df['close'].rolling(54).mean()
    loc2_chart = loc_chart > loc_chart.shift(1)
    locfiltb_chart = loc2_chart if inp3 else pd.Series(True, index=loc2_chart.index)
    locfilts_chart = ~loc2_chart if inp3 else pd.Series(True, index=loc2_chart.index)
    
    bfvg_chart = (df['low'] > df['high'].shift(2)) & volfilt_chart & atrfilt_chart & locfiltb_chart
    sfvg_chart = (df['high'] < df['low'].shift(2)) & volfilt_chart & atrfilt_chart & locfilts_chart
    
    # Reindex 4H signals to chart timeframe
    df_4h_expanded = df_4h.reindex(df_copy.index, method='ffill')
    
    # Entry conditions
    long_entry = bull_result.reindex(df_4h_expanded.index, method='ffill').fillna(False)
    short_entry = bear_fvg1_cond.reindex(df_4h_expanded.index, method='ffill').fillna(False)
    
    # Build entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i < 2:
            continue
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        if long_entry.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        
        if short_entry.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries