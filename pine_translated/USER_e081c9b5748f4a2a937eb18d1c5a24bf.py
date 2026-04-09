import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Parameters from Pine Script
    st_length = 10
    st_mult = 3.0
    wt_n1 = 10
    wt_n2 = 21
    vp_length = 20
    rsi_length = 14
    use_rsi_filter = True
    
    hlc3 = (df['high'] + df['low'] + df['close']) / 3
    
    # SuperTrend calculation (manual implementation)
    tr = np.maximum(df['high'] - df['low'], np.maximum(np.abs(df['high'] - df['close'].shift(1)), np.abs(df['low'] - df['close'].shift(1))))
    atr = tr.ewm(alpha=1/st_length, adjust=False).mean()
    hl2 = (df['high'] + df['low']) / 2
    upper_band = hl2 + st_mult * atr
    lower_band = hl2 - st_mult * atr
    close_prev = df['close'].shift(1)
    st_dir = np.zeros(len(df))
    st_bull = pd.Series(False, index=df.index)
    for i in range(1, len(df)):
        if close_prev.iloc[i] > upper_band.iloc[i]:
            st_dir[i] = 1
        elif close_prev.iloc[i] < lower_band.iloc[i]:
            st_dir[i] = -1
        else:
            st_dir[i] = st_dir[i-1]
            if st_dir[i] == 1 and df['low'].iloc[i] < lower_band.iloc[i]:
                st_dir[i] = -1
            elif st_dir[i] == -1 and df['high'].iloc[i] > upper_band.iloc[i]:
                st_dir[i] = 1
        st_bull.iloc[i] = st_dir[i] < 0
    
    # WaveTrend calculation
    esa = hlc3.ewm(span=wt_n1, adjust=False).mean()
    d_val = np.abs(hlc3 - esa).ewm(span=wt_n1, adjust=False).mean()
    ci = (hlc3 - esa) / (0.015 * d_val)
    tci = ci.ewm(span=wt_n2, adjust=False).mean()
    wt1 = tci
    wt2 = wt1.rolling(4).mean()
    
    # Volume Profile Proxy
    vol_wma = (df['close'] * df['volume']).rolling(vp_length).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True)
    vol_total = df['volume'].rolling(vp_length).mean()
    poc_proxy = vol_wma / vol_total
    
    # RSI (Wilder)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/rsi_length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # Filters
    strong_trend_long = (rsi > 60) | (rsi < 40)
    strong_trend_short = strong_trend_long
    strong_trend = strong_trend_long if use_rsi_filter else pd.Series(True, index=df.index)
    
    # Conditions
    wt_bull_cross = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    wt_bear_cross = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    
    long_condition = wt_bull_cross & st_bull & (df['close'] > poc_proxy) & strong_trend
    short_condition = wt_bear_cross & (~st_bull) & (df['close'] < poc_proxy) & strong_trend
    
    entries = []
    trade_num = 1
    in_position = False
    
    for i in range(1, len(df)):
        if not in_position:
            if long_condition.iloc[i] and not pd.isna(wt1.iloc[i]) and not pd.isna(wt2.iloc[i]) and not pd.isna(atr.iloc[i]):
                entry_price = df['close'].iloc[i]
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
                in_position = True
            elif short_condition.iloc[i] and not pd.isna(wt1.iloc[i]) and not pd.isna(wt2.iloc[i]) and not pd.isna(atr.iloc[i]):
                entry_price = df['close'].iloc[i]
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
                in_position = True
    
    return entries