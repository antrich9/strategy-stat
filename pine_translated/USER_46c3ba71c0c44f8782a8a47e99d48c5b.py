import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    pi = 2 * np.arcsin(1)
    
    # E2PSS Parameters
    period_e2pss = 15
    price_e2pss = (df['high'] + df['low']) / 2
    a1 = np.exp(-1.414 * pi / period_e2pss)
    b1 = 2 * a1 * np.cos(1.414 * pi / period_e2pss)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    # Calculate E2PSS Filt2
    filt2 = np.zeros(len(df))
    trigger_e2pss = np.zeros(len(df))
    for i in range(len(df)):
        if i < 2:
            filt2[i] = price_e2pss.iloc[i]
        else:
            filt2[i] = coef1 * price_e2pss.iloc[i] + coef2 * filt2[i-1] + coef3 * filt2[i-2]
        trigger_e2pss[i] = filt2[i-1] if i > 0 else np.nan
    
    signal_long_e2pss = filt2 > trigger_e2pss
    signal_short_e2pss = filt2 < trigger_e2pss
    use_e2pss = True
    inverse_e2pss = False
    signal_long_e2pss_final = signal_short_e2pss if inverse_e2pss else signal_long_e2pss
    signal_short_e2pss_final = signal_long_e2pss if inverse_e2pss else signal_short_e2pss
    
    # TDFI Parameters
    lookback_tdfi = 13
    mma_length_tdfi = 13
    smma_length_tdfi = 13
    n_length_tdfi = 3
    filter_high_tdfi = 0.05
    filter_low_tdfi = -0.05
    price_tdfi = df['close'] * 1000
    mma_mode_tdfi = 'ema'
    smma_mode_tdfi = 'ema'
    
    # MMA (EMA)
    mma_tdfi = price_tdfi.ewm(span=mma_length_tdfi, adjust=False).mean()
    # SMMA (Wilder smoothing)
    smma_tdfi = price_tdfi.ewm(alpha=1/smma_length_tdfi, adjust=False).mean()
    
    impet_mma_tdfi = mma_tdfi - mma_tdfi.shift(1)
    impet_smma_tdfi = smma_tdfi - smma_tdfi.shift(1)
    div_ma_tdfi = np.abs(mma_tdfi - smma_tdfi)
    aver_impet_tdfi = (impet_mma_tdfi + impet_smma_tdfi) / 2
    tdf_tdfi_raw = np.power(div_ma_tdfi, 1) * np.power(aver_impet_tdfi, n_length_tdfi)
    tdf_highest = tdf_tdfi_raw.abs().rolling(lookback_tdfi * n_length_tdfi).max()
    signal_tdfi = tdf_tdfi_raw / tdf_highest
    
    cross_tdfi = True
    use_tdfi = True
    inverse_tdfi = True
    signal_long_tdfi = (cross_tdfi and (signal_tdfi > signal_tdfi.shift(1)) & (signal_tdfi.shift(1) <= filter_high_tdfi)) if use_tdfi else pd.Series(True, index=df.index)
    signal_short_tdfi = (cross_tdfi and (signal_tdfi < signal_tdfi.shift(1)) & (signal_tdfi.shift(1) >= filter_low_tdfi)) if use_tdfi else pd.Series(True, index=df.index)
    final_long_signal_tdfi = signal_short_tdfi if inverse_tdfi else signal_long_tdfi
    final_short_signal_tdfi = signal_long_tdfi if inverse_tdfi else signal_short_tdfi
    
    # HEV Parameters
    length_hev = 200
    hv_ma_length = 20
    divisor_hev = 3.6
    
    range_1 = df['high'] - df['low']
    range_avg = range_1.rolling(length_hev).mean()
    durchschnitt = df['volume'].rolling(hv_ma_length).mean()
    volume_a = df['volume'].rolling(length_hev).mean()
    high1 = df['high'].shift(1)
    low1 = df['low'].shift(1)
    mid1 = (df['high'].shift(1) + df['low'].shift(1)) / 2
    u1 = mid1 + (high1 - low1) / divisor_hev
    d1 = mid1 - (high1 - low1) / divisor_hev
    
    r_enabled1 = (range_1 > range_avg) & (df['close'] < d1) & (df['volume'] > volume_a)
    r_enabled2 = df['close'] < mid1
    r_enabled = r_enabled1 | r_enabled2
    g_enabled1 = df['close'] > mid1
    g_enabled2 = (range_1 > range_avg) & (df['close'] > u1) & (df['volume'] > volume_a)
    g_enabled3 = (df['high'] > high1) & (range_1 < range_avg / 1.5) & (df['volume'] < volume_a)
    g_enabled4 = (df['low'] < low1) & (range_1 < range_avg / 1.5) & (df['volume'] > volume_a)
    g_enabled = g_enabled1 | g_enabled2 | g_enabled3 | g_enabled4
    
    basic_long_hev_condition = g_enabled & (df['volume'] > durchschnitt)
    basic_short_hev_condition = r_enabled & (df['volume'] > durchschnitt)
    
    use_hev = True
    highlight_movements_hev = True
    cross_hev = True
    inverse_hev = False
    
    hev_signals_long = basic_long_hev_condition if use_hev and highlight_movements_hev else g_enabled if use_hev else pd.Series(True, index=df.index)
    hev_signals_short = basic_short_hev_condition if use_hev and highlight_movements_hev else r_enabled if use_hev else pd.Series(True, index=df.index)
    hev_signals_long_cross = hev_signals_long & ~hev_signals_long.shift(1).fillna(False) if cross_hev else hev_signals_long
    hev_signals_short_cross = hev_signals_short & ~hev_signals_short.shift(1).fillna(False) if cross_hev else hev_signals_short
    hev_signals_long_final = hev_signals_short_cross if inverse_hev else hev_signals_long_cross
    hev_signals_short_final = hev_signals_long_cross if inverse_hev else hev_signals_short_cross
    
    # Entry conditions
    long_condition = signal_long_e2pss_final & final_long_signal_tdfi & basic_long_hev_condition
    short_condition = signal_short_e2pss_final & final_short_signal_tdfi & basic_short_hev_condition
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if np.isnan(mma_tdfi.iloc[i]) or np.isnan(smma_tdfi.iloc[i]) or np.isnan(range_avg.iloc[i]) or np.isnan(volume_a.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        
        if short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries