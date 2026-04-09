import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Parameters (matching Pine Script inputs)
    atrLength = 14
    atrMultiplier = 1.5
    takeProfitRatio = 1.5
    tradeDirection = "Both"
    
    lengthVidya = 14
    lengthCmo = 14
    alpha = 0.2
    
    cmoLength = 14
    cmoBuyLevel = -50
    cmoSellLevel = 50
    
    volLength = 20
    volMultiplier = 2.0
    
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    input_breakout = True
    input_retest = True
    
    startHour = 7
    endHour = 18
    
    # Calculate CMO using Wilder RSI method
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    cmo = wilder_rsi(df['close'], cmoLength)
    
    # Calculate VIDYA
    vidyaValue = np.zeros(len(df))
    vidyaValue[0] = df['close'].iloc[0]
    for i in range(1, len(df)):
        cmo_val = cmo.iloc[i] / 100.0
        vidyaValue[i] = vidyaValue[i-1] + alpha * cmo_val * (df['close'].iloc[i] - vidyaValue[i-1])
    vidyaValue = pd.Series(vidyaValue, index=df.index)
    
    # Calculate ATR using Wilder method
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/atrLength, min_periods=atrLength, adjust=False).mean()
    
    # Normalized volume
    avgVolume = df['volume'].rolling(volLength).mean()
    normVolume = df['volume'] / avgVolume
    highVolume = normVolume > volMultiplier
    
    # Pivot points (shifted to align with Pine Script indexing)
    window = 2 * input_lookback + 1
    pl_raw = df['low'].rolling(window=window, min_periods=window).min()
    ph_raw = df['high'].rolling(window=window, min_periods=window).max()
    pl = pl_raw.shift(input_lookback)
    ph = ph_raw.shift(input_lookback)
    pl = pl.ffill().bfill()
    ph = ph.ffill().bfill()
    
    pl_change = pl.diff().fillna(1.0) != 0
    ph_change = ph.diff().fillna(1.0) != 0
    
    # Box boundaries (using shifted pivots)
    pl_shift_plus_1 = pl.shift(-(input_lookback + 1))
    ph_shift_plus_1 = ph.shift(-(input_lookback + 1))
    pl_shift_minus_1 = pl.shift(-(input_lookback - 1))
    ph_shift_minus_1 = ph.shift(-(input_lookback - 1))
    
    s_yLoc = np.where(pl_shift_plus_1 > pl_shift_minus_1, pl_shift_minus_1, pl_shift_plus_1)
    r_yLoc = np.where(ph_shift_plus_1 > ph_shift_minus_1, ph_shift_plus_1, ph_shift_plus_1)
    sTop = pl
    sBot = pd.Series(s_yLoc, index=df.index)
    rTop = ph
    rBot = pd.Series(r_yLoc, index=df.index)
    
    # Crossover/crossunder for breakout detection
    co = (df['close'] > sTop) & (df['close'].shift(1) <= sTop.shift(1))
    cu = (df['close'] < rTop) & (df['close'].shift(1) >= rTop.shift(1))
    
    # barssince helper
    def barssince(cond):
        result = np.zeros(len(df))
        count = -1
        found = False
        for i in range(len(df)):
            if not found and cond.iloc[i]:
                found = True
                count = 0
            elif found:
                count += 1
            result[i] = count if found else -1
        return pd.Series(result, index=df.index)
    
    # Retest conditions
    s_break_since = barssince(co)
    r_break_since = barssince(cu)
    
    s1 = co & (s_break_since > input_retSince) & (df['high'] >= sTop) & (df['close'] <= sBot)
    s2 = co & (s_break_since > input_retSince) & (df['high'] >= sTop) & (df['close'] >= sBot) & (df['close'] <= sTop)
    s3 = co & (s_break_since > input_retSince) & (df['high'] >= sBot) & (df['high'] <= sTop)
    s4 = co & (s_break_since > input_retSince) & (df['high'] >= sBot) & (df['high'] <= sTop) & (df['close'] < sBot)
    
    r1 = cu & (r_break_since > input_retSince) & (df['low'] <= rBot) & (df['close'] >= rTop)
    r2 = cu & (r_break_since > input_retSince) & (df['low'] <= rBot) & (df['close'] <= rTop) & (df['close'] >= rBot)
    r3 = cu & (r_break_since > input_retSince) & (df['low'] <= rTop) & (df['low'] >= rBot)
    r4 = cu & (r_break_since > input_retSince) & (df['low'] <= rTop) & (df['low'] >= rBot) & (df['close'] > rTop)
    
    retestLong = s1 | s2 | s3 | s4
    retestShort = r1 | r2 | r3 | r4
    
    entries = []
    trade_num = 0
    
    for i in range(len(df)):
        if pd.isna(vidyaValue.iloc[i]) or pd.isna(avgVolume.iloc[i]):
            continue
        
        ts = df['time'].iloc[i]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        is_trading_hours = startHour <= hour < endHour
        
        if not is_trading_hours:
            continue
        
        if not highVolume.iloc[i]:
            continue
        
        if pd.isna(retestLong.iloc[i]) or pd.isna(retestShort.iloc[i]):
            continue
        
        if retestLong.iloc[i] and (tradeDirection == "Long" or tradeDirection == "Both"):
            trade_num += 1
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
        
        if retestShort.iloc[i] and (tradeDirection == "Short" or tradeDirection == "Both"):
            trade_num += 1
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
    
    return entries