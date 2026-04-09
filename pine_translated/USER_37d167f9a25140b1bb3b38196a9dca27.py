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
    # Input parameters (matching Pine Script inputs)
    i_htf = "240"
    i_rr = 2.0
    i_swing_len = 10
    i_fvg_min = 0.1
    i_body_big = 60.0
    i_wick_big = 40.0
    i_body_small = 40.0
    i_2cr = True

    # Resample to HTF timeframe
    tf_map = {"1": 1, "5": 5, "15": 15, "30": 30, "60": 60, "240": 240, "D": 1440}
    htf_minutes = tf_map.get(i_htf, 240)

    # Create a copy with datetime index for resampling
    df_work = df.copy()
    df_work['datetime'] = pd.to_datetime(df_work['time'], unit='s', utc=True)
    df_work.set_index('datetime', inplace=True)

    # Resample to HTF
    htf_tf = f'{htf_minutes}T'
    htf_df = df_work.resample(htf_tf).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'time': 'first'
    }).dropna()

    # HTF calculations
    htf_h = htf_df['high']
    htf_l = htf_df['low']
    htf_o = htf_df['open']
    htf_c = htf_df['close']
    htf_range = htf_h - htf_l
    htf_body = (htf_c - htf_o).abs()
    htf_upper_wick = htf_h - htf_c.where(htf_c > htf_o, htf_o)
    htf_lower_wick = htf_c.where(htf_c < htf_o, htf_o) - htf_l
    htf_body_pct = np.where(htf_range > 0, (htf_body / htf_range) * 100, 0)
    htf_uw_pct = np.where(htf_range > 0, (htf_upper_wick / htf_range) * 100, 0)
    htf_lw_pct = np.where(htf_range > 0, (htf_lower_wick / htf_range) * 100, 0)

    # HTF Swing Points
    def pivothigh(series, left, right):
        result = pd.Series(np.nan, index=series.index)
        for i in range(right, len(series) - left):
            if all(series.iloc[i - left:i + right + 1] <= series.iloc[i]):
                result.iloc[i] = series.iloc[i]
        return result

    def pivotlow(series, left, right):
        result = pd.Series(np.nan, index=series.index)
        for i in range(right, len(series) - left):
            if all(series.iloc[i - left:i + right + 1] >= series.iloc[i]):
                result.iloc[i] = series.iloc[i]
        return result

    htf_swing_high = pivothigh(htf_h, i_swing_len, i_swing_len)
    htf_swing_low = pivotlow(htf_l, i_swing_len, i_swing_len)

    # Fill NaN swing points with last valid value
    htf_last_sh = htf_swing_high.ffill()
    htf_last_sl = htf_swing_low.ffill()

    # HTF PCH / PCL
    htf_pch = htf_h.shift(1)
    htf_pcl = htf_l.shift(1)

    # HTF Candle Classification
    htf_near_discount = (htf_l <= htf_last_sl * 1.003) | (htf_l <= htf_pcl * 1.003)
    htf_bull_respect = (htf_lw_pct >= i_wick_big) & (htf_body_pct <= i_body_small) & (htf_c > htf_o) & htf_near_discount

    htf_near_premium = (htf_h >= htf_last_sh * 0.997) | (htf_h >= htf_pch * 0.997)
    htf_bear_respect = (htf_uw_pct >= i_wick_big) & (htf_body_pct <= i_body_small) & (htf_c < htf_o) & htf_near_premium

    htf_bull_disrespect = (htf_body_pct >= i_body_big) & (htf_c > htf_o) & ((htf_c > htf_last_sh) | (htf_c > htf_pch))
    htf_bear_disrespect = (htf_body_pct >= i_body_big) & (htf_c < htf_o) & ((htf_c < htf_last_sl) | (htf_c < htf_pcl))

    # HTF Bias
    htf_bull_bias = htf_bull_respect | htf_bull_disrespect
    htf_bear_bias = htf_bear_respect | htf_bear_disrespect

    htf_bias = pd.Series(0, index=htf_df.index)
    htf_bias_values = []
    current_bias = 0
    for i in range(len(htf_df)):
        if htf_bull_bias.iloc[i]:
            current_bias = 1
        elif htf_bear_bias.iloc[i]:
            current_bias = -1
        htf_bias_values.append(current_bias)
    htf_bias = pd.Series(htf_bias_values, index=htf_df.index)

    # LTF calculations (on input df)
    ltf_h = df['high']
    ltf_l = df['low']
    ltf_o = df['open']
    ltf_c = df['close']

    ltf_candle_range = ltf_h - ltf_l
    ltf_body = (ltf_c - ltf_o).abs()
    ltf_upper_wick = ltf_h - ltf_c.where(ltf_c > ltf_o, ltf_o)
    ltf_lower_wick = ltf_c.where(ltf_c < ltf_o, ltf_o) - ltf_l
    ltf_body_pct = np.where(ltf_candle_range > 0, (ltf_body / ltf_candle_range) * 100, 0)
    ltf_upper_wick_pct = np.where(ltf_candle_range > 0, (ltf_upper_wick / ltf_candle_range) * 100, 0)
    ltf_lower_wick_pct = np.where(ltf_candle_range > 0, (ltf_lower_wick / ltf_candle_range) * 100, 0)

    # LTF Swing Points
    ltf_swing_high = pivothigh(ltf_h, i_swing_len, i_swing_len)
    ltf_swing_low = pivotlow(ltf_l, i_swing_len, i_swing_len)

    ltf_last_sh = ltf_swing_high.ffill()
    ltf_last_sl = ltf_swing_low.ffill()

    pch = ltf_h.shift(1)
    pcl = ltf_l.shift(1)

    # FVG Detection
    bull_fvg = (ltf_l > ltf_h.shift(2)) & ((ltf_l - ltf_h.shift(2)) / ltf_c * 100 >= i_fvg_min)
    bear_fvg = (ltf_h < ltf_l.shift(2)) & ((ltf_l.shift(2) - ltf_h) / ltf_c * 100 >= i_fvg_min)

    bull_fvg_top = pd.Series(np.nan, index=df.index)
    bull_fvg_bot = pd.Series(np.nan, index=df.index)
    bear_fvg_top = pd.Series(np.nan, index=df.index)
    bear_fvg_bot = pd.Series(np.nan, index=df.index)

    bull_fvg_top_prev = np.nan
    bull_fvg_bot_prev = np.nan
    bear_fvg_top_prev = np.nan
    bear_fvg_bot_prev = np.nan

    for i in range(len(df)):
        if bull_fvg.iloc[i]:
            bull_fvg_top_prev = ltf_l.iloc[i]
            bull_fvg_bot_prev = ltf_h.iloc[i]
        bull_fvg_top.iloc[i] = bull_fvg_top_prev
        bull_fvg_bot.iloc[i] = bull_fvg_bot_prev

        if bear_fvg.iloc[i]:
            bear_fvg_top_prev = ltf_l.iloc[i]
            bear_fvg_bot_prev = ltf_h.iloc[i]
        bear_fvg_top.iloc[i] = bear_fvg_top_prev
        bear_fvg_bot.iloc[i] = bear_fvg_bot_prev

    # LTF Respect Candles
    ltf_near_discount = (ltf_l <= ltf_last_sl * 1.002) | (ltf_l <= pcl * 1.002) | ((ltf_l <= bull_fvg_top) & (ltf_l >= bull_fvg_bot))
    ltf_near_premium = (ltf_h >= ltf_last_sh * 0.998) | (ltf_h >= pch * 0.998) | ((ltf_h <= bear_fvg_top) & (ltf_h >= bear_fvg_bot))

    ltf_last_sl_filled = ltf_last_sl.fillna(0)
    ltf_last_sh_filled = ltf_last_sh.fillna(999999)

    bull_respect = (ltf_lower_wick_pct >= i_wick_big) & (ltf_body_pct <= i_body_small) & (ltf_c > ltf_o) & ltf_near_discount & (ltf_c > np.maximum(ltf_last_sl_filled, pcl))
    bear_respect = (ltf_upper_wick_pct >= i_wick_big) & (ltf_body_pct <= i_body_small) & (ltf_c < ltf_o) & ltf_near_premium & (ltf_c < np.minimum(ltf_last_sh_filled, pch))

    # LTF Disrespect Candles
    bull_disrespect = (ltf_body_pct >= i_body_big) & (ltf_c > ltf_o) & ((ltf_c > ltf_last_sh_filled) | (ltf_c > pch) | ((~bear_fvg_top.isna()) & (ltf_c > bear_fvg_top)))
    bear_disrespect = (ltf_body_pct >= i_body_big) & (ltf_c < ltf_o) & ((ltf_c < ltf_last_sl_filled) | (ltf_c < pcl) | ((~bull_fvg_bot.isna()) & (ltf_c < bull_fvg_bot)))

    # 2CR Filter
    bull_2cr_ok = bull_respect | (i_2cr & bear_disrespect.shift(1) & bull_respect)
    bear_2cr_ok = bear_respect | (i_2cr & bull_disrespect.shift(1) & bear_respect)

    # Map HTF bias to LTF timeframe
    htf_bias_ltf = pd.Series(0, index=df.index)
    for i in range(len(df)):
        ltf_ts = df['time'].iloc[i]
        htf_idx = htf_bias.index[np.searchsorted(htf_bias.index.values, pd.Timestamp(ltf_ts, tz='UTC'))]
        if htf_idx in htf_bias.index:
            htf_bias_ltf.iloc[i] = htf_bias.loc[htf_idx]

    # Entry Signals
    long_signal = (bull_2cr_ok | bull_disrespect) & (htf_bias_ltf == 1)
    short_signal = (bear_2cr_ok | bear_disrespect) & (htf_bias_ltf == -1)

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(long_signal.iloc[i]) or pd.isna(short_signal.iloc[i]):
            continue

        if long_signal.iloc[i]:
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
            trade_num += 1
        elif short_signal.iloc[i]:
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
            trade_num += 1

    return entries