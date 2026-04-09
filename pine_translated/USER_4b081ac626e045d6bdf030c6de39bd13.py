import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']
    open_col = df['open']
    
    entries = []
    trade_num = 1
    
    # EMA 200
    ema200 = close.ewm(span=200, adjust=False).mean()
    
    # Trend conditions
    trendBullish = close > ema200
    trendBearish = close < ema200
    
    # DMI ADX (14, 14) - Wilder implementation
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    atr_len = 14
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    atr = tr.ewm(alpha=1/atr_len, adjust=False).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/atr_len, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/atr_len, adjust=False).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/atr_len, adjust=False).mean()
    strongTrend = adx > 25
    
    # Swing Highs/Lows
    is_swing_high = (high.shift(1) < high) & (high.shift(2) < high.shift(1))
    is_swing_low = (low.shift(1) > low) & (low.shift(2) > low.shift(1))
    
    # RSI (Wilder)
    rsi_len = 14
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/rsi_len, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_len, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsiBullish = rsi < 30
    rsiBearish = rsi > 70
    
    # Liquidity sweeps
    prev_day_high = high.shift(1).rolling(1).max().shift(1)
    prev_day_low = low.shift(1).rolling(1).min().shift(1)
    swept_high = high > prev_day_high
    swept_low = low < prev_day_low
    
    for i in range(len(df)):
        if i < 2:
            continue
        if pd.isna(ema200.iloc[i]) or pd.isna(rsi.iloc[i]):
            continue
        
        bullish_conditions = (trendBullish.iloc[i] and 
                             strongTrend.iloc[i] and 
                             (is_swing_low.iloc[i] or swept_low.iloc[i]) and 
                             rsiBullish.iloc[i])
        
        bearish_conditions = (trendBearish.iloc[i] and 
                             strongTrend.iloc[i] and 
                             (is_swing_high.iloc[i] or swept_high.iloc[i]) and 
                             rsiBearish.iloc[i])
        
        if bullish_conditions:
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if bearish_conditions:
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries