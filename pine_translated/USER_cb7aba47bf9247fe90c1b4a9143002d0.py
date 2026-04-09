import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    def alma(data, length, offset, sigma):
        m = np.floor(offset * (length - 1))
        s = length / sigma
        w = np.exp(-((np.arange(length) - m) ** 2) / (2 * s * s))
        return np.convolve(data, w / w.sum(), mode='valid')
    
    def wilder_rsi(src, length):
        delta = src.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def wilder_atr(high, low, close, length):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        return atr
    
    min_body_pct = 70
    search_factor = 1.3
    ma_length = 20
    ma_length_b = 8
    ma_reaction = 1
    ma_reaction_b = 1
    previous_bars_count = 100
    
    atr = wilder_atr(df['high'], df['low'], df['close'], previous_bars_count)
    atr_prev = atr.shift(1)
    
    is_bullish = df['close'] > df['open']
    is_bearish = df['close'] < df['open']
    body = (df['close'] - df['open']).abs()
    body_pct = body / (df['high'] - df['low']) * 100
    
    is_valid_bullish = is_bullish & (body_pct >= min_body_pct)
    is_valid_bearish = is_bearish & (body_pct >= min_body_pct)
    is_strong_bullish = is_valid_bullish & (body >= atr_prev * search_factor)
    is_strong_bearish = is_valid_bearish & (body >= atr_prev * search_factor)
    
    slow_ma = df['close'].rolling(window=ma_length).mean()
    fast_ma = df['close'].rolling(window=ma_length_b).mean()
    
    slow_ma_trend = np.where(slow_ma > slow_ma.shift(ma_reaction), 1, np.where(slow_ma < slow_ma.shift(ma_reaction), -1, 0))
    fast_ma_trend = np.where(fast_ma > fast_ma.shift(ma_reaction_b), 1, np.where(fast_ma < fast_ma.shift(ma_reaction_b), -1, 0))
    
    is_price_above_fast_ma = df['close'] > fast_ma
    is_price_above_slow_ma = df['close'] > slow_ma
    is_price_above_both_ma = is_price_above_slow_ma & is_price_above_fast_ma
    is_price_above_slow_ma_with_bullish = is_price_above_slow_ma & (slow_ma_trend > 0)
    is_price_above_fast_ma_with_bullish = is_price_above_fast_ma & (fast_ma_trend > 0)
    is_price_above_both_ma_with_bullish = is_price_above_both_ma & (slow_ma_trend > 0) & (fast_ma_trend > 0)
    is_slow_ma_trend_bullish = slow_ma_trend > 0
    is_fast_ma_trend_bullish = fast_ma_trend > 0
    is_both_ma_trend_bullish = (slow_ma_trend > 0) & (fast_ma_trend > 0)
    no_bullish_condition = True
    
    is_price_below_fast_ma = df['close'] < fast_ma
    is_price_below_slow_ma = df['close'] < slow_ma
    is_price_below_both_ma = is_price_below_slow_ma & is_price_below_fast_ma
    is_price_below_slow_ma_with_bearish = is_price_below_slow_ma & (slow_ma_trend < 0)
    is_price_below_fast_ma_with_bearish = is_price_below_fast_ma & (fast_ma_trend < 0)
    is_price_below_both_ma_with_bearish = is_price_below_both_ma & (slow_ma_trend < 0) & (fast_ma_trend < 0)
    is_slow_ma_trend_bearish = slow_ma_trend < 0
    is_fast_ma_trend_bearish = fast_ma_trend < 0
    is_both_ma_trend_bearish = (slow_ma_trend < 0) & (fast_ma_trend < 0)
    no_bearish_condition = True
    
    filter_type = "CON FILTRADO DE TENDENCIA"
    
    final_green_elephant = is_strong_bullish & (
        is_price_above_fast_ma | is_price_above_slow_ma | is_price_above_both_ma |
        is_price_above_slow_ma_with_bullish | is_price_above_fast_ma_with_bullish |
        is_price_above_both_ma_with_bullish |
        is_slow_ma_trend_bullish | is_fast_ma_trend_bullish | is_both_ma_trend_bullish |
        no_bullish_condition
    )
    
    final_red_elephant = is_strong_bearish & (
        is_price_below_fast_ma | is_price_below_slow_ma | is_price_below_both_ma |
        is_price_below_slow_ma_with_bearish | is_price_below_fast_ma_with_bearish |
        is_price_below_both_ma_with_bearish |
        is_slow_ma_trend_bearish | is_fast_ma_trend_bearish | is_both_ma_trend_bearish |
        no_bearish_condition
    )
    
    result_green = final_green_elephant & True
    result_red = final_red_elephant & True
    
    smooth_trendilo = 1
    length_trendilo = 50
    offset_trendilo = 0.85
    sigma_trendilo = 6
    bmult_trendilo = 1.0
    
    src = df['close']
    pch = src.diff(smooth_trendilo) / src * 100
    pch_filled = pch.fillna(0).values
    blength = length_trendilo
    
    avpch_values = alma(pch_filled, blength, offset_trendilo, sigma_trendilo)
    avpch = pd.Series(np.concatenate([np.array([np.nan] * (blength - 1)), avpch_values]), index=df.index)
    
    rms_values = bmult_trendilo * np.sqrt(pd.Series(avpch_values ** 2).rolling(blength).mean().values)
    rms = pd.Series(np.concatenate([np.array([np.nan] * (blength - 1)), rms_values]), index=df.index)
    
    cdir = np.where(avpch > rms, 1, np.where(avpch < -rms, -1, 0))
    cdir_series = pd.Series(np.concatenate([np.array([np.nan] * (blength - 1)), cdir]), index=df.index)
    
    min_idx = max(blength, ma_length, ma_length_b)
    
    entries = []
    trade_num = 1
    
    for i in range(min_idx, len(df)):
        if pd.isna(result_green.iloc[i]) or pd.isna(result_red.iloc[i]) or pd.isna(cdir_series.iloc[i]):
            continue
        if result_green.iloc[i] and cdir_series.iloc[i] > 0:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif result_red.iloc[i] and cdir_series.iloc[i] < 0:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries