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
    
    # Strategy parameters
    atrLength = 14
    atrMultiplier = 2.5
    lengthMD = 10
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate ATR (Wilder ATR)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atrLength, adjust=False).mean()
    
    # McGinley Dynamic
    md = close.copy().astype(float)
    md.iloc[0] = close.iloc[0]
    k = lengthMD
    for i in range(1, len(md)):
        if not pd.isna(md.iloc[i-1]) and not pd.isna(close.iloc[i]):
            ratio = close.iloc[i] / md.iloc[i-1]
            md.iloc[i] = md.iloc[i-1] + (close.iloc[i] - md.iloc[i-1]) / (k * (ratio ** 4))
    
    # Pivot detection using rolling min/max
    bb = input_lookback
    
    pl = low.rolling(window=bb+1, min_periods=bb+1).min().shift(1)
    ph = high.rolling(window=bb+1, min_periods=bb+1).max().shift(1)
    
    # Fixnan for pivots
    pl = pl.bfill()
    ph = ph.bfill()
    
    # Box boundaries for support
    s_yLoc = pd.where(low.shift(bb + 1) > low.shift(bb - 1), low.shift(bb - 1), low.shift(bb + 1))
    s_yLoc = pd.Series(s_yLoc, index=low.index)
    sBot = pd.Series(np.where(pl.shift(1) > pl.shift(-1), low.shift(bb - 1), low.shift(bb + 1)), index=low.index)
    
    # Box boundaries for resistance
    r_yLoc = pd.where(high.shift(bb + 1) > high.shift(bb - 1), high.shift(bb + 1), high.shift(bb - 1))
    r_yLoc = pd.Series(r_yLoc, index=high.index)
    rTop = pd.Series(np.where(ph.shift(1) > ph.shift(-1), high.shift(bb + 1), high.shift(bb - 1)), index=high.index)
    
    # Support/Resistance boxes (using pivots as reference)
    sBot = pl.bfill()
    sTop = pl.bfill()
    rBot = ph.bfill()
    rTop = ph.bfill()
    
    # Change detection for pivots
    pl_change = (pl != pl.shift(1)) & pl.notna()
    ph_change = (ph != ph.shift(1)) & ph.notna()
    
    # ATR-based stop loss and take profit (for reference, not used in entries)
    stopLossLong = close - (atr * atrMultiplier)
    stopLossShort = close + (atr * atrMultiplier)
    takeProfitLong = close + ((atr * atrMultiplier) * 3)
    takeProfitShort = close - ((atr * atrMultiplier) * 3)
    
    # Forecast Oscillator (calculated but may not be used in entries)
    foLength = 14
    forecastPrice = close.rolling(window=foLength).apply(lambda x: np.polyfit(range(foLength), x, 1)[0] * (foLength - 1) + np.polyfit(range(foLength), x, 1)[1], raw=True)
    forecastOsc = (close - forecastPrice) / close * 100
    
    # Breakout detection (non-repainting)
    sBreak = pd.Series(False, index=close.index)
    rBreak = pd.Series(False, index=close.index)
    
    cu = (close < sBot) & (close.shift(1) >= sBot.shift(1)) & sBot.notna()
    co = (close > rTop) & (close.shift(1) <= rTop.shift(1)) & rTop.notna()
    
    # Update sBreak and rBreak states
    for i in range(1, len(close)):
        if cu.iloc[i] and pd.isna(sBreak.iloc[i-1]):
            sBreak.iloc[i] = True
        elif pl_change.iloc[i]:
            sBreak.iloc[i] = False
        else:
            sBreak.iloc[i] = sBreak.iloc[i-1] if i > 0 else False
            
        if co.iloc[i] and pd.isna(rBreak.iloc[i-1]):
            rBreak.iloc[i] = True
        elif ph_change.iloc[i]:
            rBreak.iloc[i] = False
        else:
            rBreak.iloc[i] = rBreak.iloc[i-1] if i > 0 else False
    
    sBreak = sBreak.fillna(False)
    rBreak = rBreak.fillna(False)
    
    # Retest conditions for support (short entries)
    s1 = sBreak & (high >= sTop) & (close <= sBot)
    s2 = sBreak & (high >= sTop) & (close >= sBot) & (close <= sTop)
    s3 = sBreak & (high >= sBot) & (high <= sTop)
    s4 = sBreak & (high >= sBot) & (high <= sTop) & (close < sBot)
    
    s_ret_conditions = s1 | s2 | s3 | s4
    
    # Retest conditions for resistance (long entries)
    r1 = rBreak & (low <= rBot) & (close >= rTop)
    r2 = rBreak & (low <= rBot) & (close <= rTop) & (close >= rBot)
    r3 = rBreak & (low <= rTop) & (low >= rBot)
    r4 = rBreak & (low <= rTop) & (low >= rBot) & (close > rTop)
    
    r_ret_conditions = r1 | r2 | r3 | r4
    
    # Calculate barssince for retests
    def barssince(series):
        result = pd.Series(-1, index=series.index)
        count = -1
        for i in range(len(series)):
            if series.iloc[i]:
                count = 0
            elif count >= 0:
                count += 1
            result.iloc[i] = count
        return result
    
    s_ret_since = barssince(s_ret_conditions)
    r_ret_since = barssince(r_ret_conditions)
    
    # Support retest valid (bars since breakout > retSince and within retValid)
    sRetValid = (s_ret_since > input_retSince) & (s_ret_since <= input_retValid) & s_ret_conditions
    sRetValid = sRetValid.fillna(False)
    
    # Resistance retest valid
    rRetValid = (r_ret_since > input_retSince) & (r_ret_since <= input_retValid) & r_ret_conditions
    rRetValid = rRetValid.fillna(False)
    
    # Entry signals
    long_entry = (co & rBreak) | rRetValid
    short_entry = (cu & sBreak) | sRetValid
    
    # Generate entry list
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_entry.iloc[i] and not pd.isna(close.iloc[i]):
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
        elif short_entry.iloc[i] and not pd.isna(close.iloc[i]):
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
    
    return entries