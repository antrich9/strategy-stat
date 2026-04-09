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
    
    # Parameters
    bb = 20  # lookback
    input_retSince = 2  # bars since breakout
    input_retValid = 2  # retest detection limiter
    rTon = True  # repaint On
    
    # Calculate CMO
    def calc_cmo(src, length):
        mom = src.diff()
        gain = mom.copy()
        loss = mom.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        cmo = 100 - (100 / (1 + rs))
        return cmo
    
    cmoLength = 14
    cmo = calc_cmo(df['close'], cmoLength)
    
    # Calculate VIDYA
    alpha = 0.2
    lengthCmo = 14
    cmo2_vals = calc_cmo(df['close'], lengthCmo)
    
    vidya = np.zeros(len(df))
    vidya[0] = df['close'].iloc[0]
    for i in range(1, len(df)):
        vidya[i] = vidya[i-1] + alpha * (cmo2_vals.iloc[i] / 100) * (df['close'].iloc[i] - vidya[i-1])
    
    # Calculate pivot points
    pl = np.where(pd.Series(df['low']).rolling(window=bb+1, min_periods=bb+1).min().shift(-bb) == df['low'], df['low'], np.nan)
    ph = np.where(pd.Series(df['high']).rolling(window=bb+1, min_periods=bb+1).max().shift(-bb) == df['high'], df['high'], np.nan)
    
    # Calculate support and resistance boxes
    pl_filled = pd.Series(pl).ffill()
    ph_filled = pd.Series(ph).ffill()
    
    s_yLoc = np.where(df['low'].shift(bb+1) > df['low'].shift(bb-1), df['low'].shift(bb-1), df['low'].shift(bb+1))
    r_yLoc = np.where(df['high'].shift(bb+1) > df['high'].shift(bb-1), df['high'].shift(bb+1), df['high'].shift(bb-1))
    
    sTop = pl_filled
    sBot = s_yLoc
    rTop = r_yLoc
    rBot = ph_filled
    
    # Calculate breakout conditions
    cu = (df['close'] < sBot) & (df['close'].shift(1) >= sBot.shift(1))
    co = (df['close'] > rTop) & (df['close'].shift(1) <= rTop.shift(1))
    
    # Calculate retest conditions
    s1 = cu & (df['high'] >= sTop) & (df['close'] <= sBot)
    s2 = cu & (df['high'] >= sTop) & (df['close'] >= sBot) & (df['close'] <= sTop)
    s3 = cu & (df['high'] >= sBot) & (df['high'] <= sTop)
    s4 = cu & (df['high'] >= sBot) & (df['high'] <= sTop) & (df['close'] < sBot)
    
    r1 = co & (df['low'] <= rBot) & (df['close'] >= rTop)
    r2 = co & (df['low'] <= rBot) & (df['close'] <= rTop) & (df['close'] >= rBot)
    r3 = co & (df['low'] <= rTop) & (df['low'] >= rBot)
    r4 = co & (df['low'] <= rTop) & (df['low'] >= rBot) & (df['close'] > rTop)
    
    sRetValid = s1 | s2 | s3 | s4
    rRetValid = r1 | r2 | r3 | r4
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.notna(sRetValid.iloc[i]) and sRetValid.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i]/1000, tz=timezone.utc).isoformat(),
                'entry_price': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif pd.notna(rRetValid.iloc[i]) and rRetValid.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i]/1000, tz=timezone.utc).isoformat(),
                'entry_price': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries