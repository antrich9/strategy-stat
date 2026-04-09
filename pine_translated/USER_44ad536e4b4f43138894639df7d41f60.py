import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['total_minutes'] = df['hour'] * 60 + df['minute']
    
    # Session definitions (in minutes from midnight)
    box_start = 7 * 60 + 45  # 0745
    box_end = 8 * 60          # 0800
    trade_start = 8 * 60      # 0800
    trade_end = 17 * 60        # 1700
    
    # Session booleans
    in_box_session = (df['total_minutes'] >= box_start) & (df['total_minutes'] < box_end)
    in_trade_session = (df['total_minutes'] >= trade_start) & (df['total_minutes'] < trade_end)
    
    # Box detection
    in_box_session_prev = in_box_session.shift(1).fillna(False)
    box_started = in_box_session & ~in_box_session_prev
    box_ended = ~in_box_session & in_box_session_prev
    
    # New day detection
    new_day = df['datetime'].dt.date != df['datetime'].dt.date.shift(1)
    new_day.iloc[0] = True
    
    # ATR calculation (Wilder)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/14, min_periods=14, adjust=False).mean()
    
    # EMA calculations
    ema_slow = df['close'].ewm(span=200, adjust=False).mean()
    ema_fast = df['close'].ewm(span=50, adjust=False).mean()
    up_trend = ema_fast > ema_slow
    down_trend = ema_fast < ema_slow
    
    # Tracking variables
    tmp_high = np.nan
    tmp_low = np.nan
    box_high = np.nan
    box_low = np.nan
    
    eq_high = 10000.0
    day_start_net = np.nan
    block_new_trades = False
    
    # Settings
    use_session_filt = True
    use_trend_filter = True
    allow_longs = True
    allow_shorts = True
    use_daily_loss_guard = True
    max_daily_loss = 300.0
    use_max_dd_guard = False
    max_dd_pct = 8.0
    atr_mult_sl = 1.5
    
    entries = []
    trade_num = 1
    last_trade_idx = -1
    
    for i in range(len(df)):
        if pd.isna(df['close'].iloc[i]) or pd.isna(ema_fast.iloc[i]) or pd.isna(ema_slow.iloc[i]) or pd.isna(atr.iloc[i]):
            continue
        
        if new_day.iloc[i]:
            day_start_net = 0.0
            block_new_trades = False
            tmp_high = np.nan
            tmp_low = np.nan
        
        curr_eq = 10000.0
        
        if use_daily_loss_guard and not pd.isna(day_start_net):
            curr_day_pl = 0.0 - day_start_net
            if curr_day_pl <= -max_daily_loss:
                block_new_trades = True
        
        if use_max_dd_guard and eq_high > 0:
            curr_dd_pct = 100.0 * (eq_high - curr_eq) / eq_high
            if curr_dd_pct >= max_dd_pct:
                block_new_trades = True
        
        if box_started.iloc[i]:
            tmp_high = df['high'].iloc[i]
            tmp_low = df['low'].iloc[i]
        
        if in_box_session.iloc[i]:
            if pd.isna(tmp_high):
                tmp_high = df['high'].iloc[i]
                tmp_low = df['low'].iloc[i]
            else:
                tmp_high = max(tmp_high, df['high'].iloc[i])
                tmp_low = min(tmp_low, df['low'].iloc[i])
        
        if box_ended.iloc[i]:
            box_high = tmp_high
            box_low = tmp_low
        
        can_trade = (not use_session_filt or in_trade_session.iloc[i]) and not block_new_trades and not pd.isna(box_high) and not in_box_session.iloc[i]
        
        sl_dist = atr_mult_sl * atr.iloc[i]
        qty_ok_long = sl_dist > 0
        qty_ok_short = sl_dist > 0
        
        long_cond = can_trade and allow_longs and df['close'].iloc[i] > box_high and (not use_trend_filter or up_trend.iloc[i]) and qty_ok_long
        short_cond = can_trade and allow_shorts and df['close'].iloc[i] < box_low and (not use_trend_filter or down_trend.iloc[i]) and qty_ok_short
        
        if long_cond and i != last_trade_idx:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': df['datetime'].iloc[i].isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(box_high),
                'raw_price_b': float(box_high)
            })
            trade_num += 1
            last_trade_idx = i
        
        if short_cond and i != last_trade_idx:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': df['datetime'].iloc[i].isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(box_low),
                'raw_price_b': float(box_low)
            })
            trade_num += 1
            last_trade_idx = i
    
    return entries