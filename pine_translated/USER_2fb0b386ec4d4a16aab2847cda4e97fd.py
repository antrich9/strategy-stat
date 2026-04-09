import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    entries = []
    trade_num = 0
    
    # Create datetime from timestamp
    dt = pd.to_datetime(df['time'], unit='s', utc=True)
    hours = dt.dt.hour
    
    # Asia session: 23:00 to 07:00
    in_asia_session = (hours >= 23) | (hours < 7)
    
    # Session high/low tracking
    asia_session_high = np.nan * np.zeros(len(df))
    asia_session_low = np.nan * np.zeros(len(df))
    
    session_open_high = np.nan
    session_open_low = np.nan
    
    for i in range(len(df)):
        if i > 0:
            prev_in_session = in_asia_session.iloc[i-1]
        else:
            prev_in_session = False
            
        current_in_session = in_asia_session.iloc[i]
        
        if current_in_session and not prev_in_session:
            session_open_high = df['high'].iloc[i]
            session_open_low = df['low'].iloc[i]
        elif current_in_session:
            session_open_high = max(session_open_high, df['high'].iloc[i])
            session_open_low = min(session_open_low, df['low'].iloc[i])
        
        if not current_in_session and prev_in_session:
            asia_session_high[i] = session_open_high
            asia_session_low[i] = session_open_low
        elif not current_in_session:
            asia_session_high[i] = asia_session_high[i-1] if i > 0 else np.nan
            asia_session_low[i] = asia_session_low[i-1] if i > 0 else np.nan
    
    # Indicators
    sma_close = df['close'].rolling(54).mean()
    
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def wilder_atr(length):
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    
    atr = wilder_atr(20) / 1.5
    vol_sma = df['volume'].rolling(9).mean()
    
    locfiltb = sma_close > sma_close.shift(1)
    locfilts = sma_close <= sma_close.shift(1)
    volfilt = df['volume'] > vol_sma * 1.5
    atrfilt = (df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr)
    
    # FVG conditions
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    prev_day_high = np.nan
    prev_day_low = np.nan
    
    consecutive_bfvg = 0
    consecutive_sfvg = 0
    flag_bfvg = False
    
    for i in range(2, len(df)):
        if pd.isna(sma_close.iloc[i]):
            continue
            
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        current_close = df['close'].iloc[i]
        
        if not in_asia_session.iloc[i] and in_asia_session.iloc[i-1]:
            prev_day_high = current_high
            prev_day_low = current_low
        elif not in_asia_session.iloc[i]:
            if not pd.isna(asia_session_high.iloc[i]):
                prev_day_high = asia_session_high.iloc[i]
            if not pd.isna(asia_session_low.iloc[i]):
                prev_day_low = asia_session_low.iloc[i]
        
        if bfvg.iloc[i]:
            consecutive_bfvg += 1
            flag_bfvg = True
            consecutive_sfvg = 0
        elif sfvg.iloc[i]:
            consecutive_sfvg += 1
            if flag_bfvg and consecutive_sfvg == 1:
                flag_bfvg = False
            flag_bfvg = False
        else:
            consecutive_bfvg = 0
            consecutive_sfvg = 0
            flag_bfvg = False
        
        bullish_sweep_high = not pd.isna(prev_day_high) and current_low <= prev_day_low and current_close > prev_day_high
        bearish_sweep_low = not pd.isna(prev_day_low) and current_high >= prev_day_high and current_close < prev_day_low
        
        entry_long = bfvg.iloc[i] and bullish_sweep_high
        entry_short = sfvg.iloc[i] and bearish_sweep_low
        
        if entry_long:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entry_price = float(current_close)
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
        elif entry_short:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entry_price = float(current_close)
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
    
    return entries