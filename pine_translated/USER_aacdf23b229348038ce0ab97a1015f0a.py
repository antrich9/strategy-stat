import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    ts = df['time']
    
    t3_fast_length = 12
    t3_slow_length = 25
    t3_factor = 0.7
    adx_length = 14
    di_length = 14
    adx_threshold = 25.0
    bb_length = 20
    bb_mult = 2.0
    keltner_length = 20
    keltner_mult = 1.5
    trade_style = "Balanced"
    entry_filter = 3
    
    def adx_calc(high, low, close, adx_len, di_len):
        prev_close = close.shift(1)
        high_diff = high - high.shift(1)
        low_diff = low - low.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        up = high_diff
        down = low_diff
        plus_dm = pd.Series(0.0, index=low.index)
        minus_dm = pd.Series(0.0, index=low.index)
        mask_plus = (up > down) & (up > 0)
        mask_minus = (down > up) & (down > 0)
        plus_dm[mask_plus] = up[mask_plus]
        minus_dm[mask_minus] = down[mask_minus]
        alpha_di = 1.0 / di_len
        rma_tr = tr.ewm(alpha=alpha_di, min_periods=di_len, adjust=False).mean()
        rma_plus_dm_vals = plus_dm.ewm(alpha=alpha_di, min_periods=di_len, adjust=False).mean()
        rma_minus_dm_vals = minus_dm.ewm(alpha=alpha_di, min_periods=di_len, adjust=False).mean()
        tr_filled = rma_tr.replace(0, 1)
        plus_di = (rma_plus_dm_vals / tr_filled) * 100
        minus_di = (rma_minus_dm_vals / tr_filled) * 100
        sum_di = plus_di + minus_di
        adx_val = (abs(plus_di - minus_di) / sum_di.replace(0, 1)) * 100
        alpha_adx = 1.0 / adx_len
        adx = adx_val.ewm(alpha=alpha_adx, min_periods=adx_len, adjust=False).mean()
        return adx, plus_di, minus_di
    
    def t3_calc(src, length, factor):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        ema4 = ema3.ewm(span=length, adjust=False).mean()
        ema5 = ema4.ewm(span=length, adjust=False).mean()
        ema6 = ema5.ewm(span=length, adjust=False).mean()
        c1 = -factor**3
        c2 = 3 * factor**2 + 3 * factor**3
        c3 = -6 * factor**2 - 3 * factor - 3 * factor**3
        c4 = 1 + 3 * factor + factor**3 + 3 * factor**2
        t3 = c1 * ema6 + c2 * ema5 + c3 * ema4 + c4 * ema3
        return t3
    
    t3_fast = t3_calc(close, t3_fast_length, t3_factor)
    t3_slow = t3_calc(close, t3_slow_length, t3_factor)
    adx_value, plus_di, minus_di = adx_calc(high, low, close, adx_length, di_length)
    
    keltner_basis = close.ewm(span=keltner_length, adjust=False).mean()
    prev_close = close.shift(1).fillna(open_price)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_keltner = tr.ewm(alpha=1/keltner_length, min_periods=keltner_length, adjust=False).mean()
    keltner_upper = keltner_basis + keltner_mult * atr_keltner
    keltner_lower = keltner_basis - keltner_mult * atr_keltner
    
    bb_basis = close.rolling(bb_length).mean()
    bb_std = close.rolling(bb_length).std()
    bb_upper = bb_basis + bb_mult * bb_std
    bb_lower = bb_basis - bb_mult * bb_std
    
    cross_up = (t3_fast > t3_slow) & (t3_fast.shift(1) <= t3_slow.shift(1))
    cross_down = (t3_fast < t3_slow) & (t3_fast.shift(1) >= t3_slow.shift(1))
    bullish_breakout = (bb_upper < keltner_upper) & (bb_upper.shift(1) >= keltner_upper.shift(1))
    bearish_breakout = (bb_lower > keltner_lower) & (bb_lower.shift(1) <= keltner_lower.shift(1))
    adx_confirm = adx_value > adx_threshold
    
    raw_long_signal = (t3_fast > t3_slow) & (close > t3_slow) & bullish_breakout & adx_confirm
    raw_short_signal = (t3_fast < t3_slow) & (close < t3_slow) & bearish_breakout & adx_confirm
    
    in_long_position = False
    in_short_position = False
    bars_since_entry = 1000
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if raw_long_signal.iloc[i] and bars_since_entry >= entry_filter and not in_long_position:
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts.iloc[i]),
                'entry_time': datetime.fromtimestamp(ts.iloc[i] / 1000 if ts.iloc[i] > 9999999999 else ts.iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            in_long_position = True
            in_short_position = False
            bars_since_entry = 0
        elif raw_short_signal.iloc[i] and bars_since_entry >= entry_filter and not in_short_position:
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts.iloc[i]),
                'entry_time': datetime.fromtimestamp(ts.iloc[i] / 1000 if ts.iloc[i] > 9999999999 else ts.iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            in_short_position = True
            in_long_position = False
            bars_since_entry = 0
        else:
            bars_since_entry += 1
    
    return entries