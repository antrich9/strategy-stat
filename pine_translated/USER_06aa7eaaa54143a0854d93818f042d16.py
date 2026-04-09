import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Parameters (from Pine Script inputs)
    alma_window = 9
    alma_offset = 0.85
    alma_sigma = 6.0
    
    flux_smooth = 1
    flux_length = 50
    flux_offset = 0.85
    flux_sigma = 6
    
    qs_period = 20
    qs_deviation = 2.0
    
    vol_length = 50
    vol_high = 150
    vol_normal = 75
    
    entry_lookback = 5
    entry_threshold = 0.5
    use_alma_signal = True
    use_flux_signal = True
    use_qs_signal = True
    use_vol_filter = True

    # ALMA calculation
    def calc_alma(src, window, offset, sigma):
        m = offset * (window - 1)
        s = sigma * (window - 1) / sigma
        w = np.exp(-np.pow(np.arange(window) - m, 2) / (2 * s * s))
        w = w / w.sum()
        result = np.convolve(src, w, mode='valid')
        padded = np.concatenate([result, np.full(window - 1, np.nan)])
        return pd.Series(padded, index=src.index)

    df['alma'] = calc_alma(df['close'].values, alma_window, alma_offset, alma_sigma)

    # FluxWave Oscillator
    alma_flux = calc_alma(df['close'].values, flux_length, flux_offset, flux_sigma)
    momentum = df['close'] - alma_flux
    df['flux'] = calc_alma(momentum.values, flux_smooth, flux_offset, flux_sigma)
    df['flux_ma'] = calc_alma(df['flux'].values, 9, 0.85, 6)

    # QuickSilver Bands
    qs_middle = df['close'].rolling(qs_period).mean()
    qs_std = df['close'].rolling(qs_period).std()
    df['qs_upper'] = qs_middle + (qs_std * qs_deviation)
    df['qs_lower'] = qs_middle - (qs_std * qs_deviation)

    # Volume Energy (Normalized)
    vol_ma = df['volume'].rolling(vol_length).mean()
    vol_std = df['volume'].rolling(vol_length).std()
    df['vol_energy'] = (df['volume'] - vol_ma) / vol_std * 100

    # Build condition Series
    alma_bullish = (df['close'] > df['alma']) & (df['close'].shift(1) <= df['alma'].shift(1))
    alma_bearish = (df['close'] < df['alma']) & (df['close'].shift(1) >= df['alma'].shift(1))

    flux_std = df['flux_ma'].std()
    flux_threshold = flux_std * flux_band_mult

    flux_bullish = df['flux'] > flux_threshold
    flux_bearish = df['flux'] < -flux_threshold

    qs_bullish = df['close'] > df['qs_upper']
    qs_bearish = df['close'] < df['qs_lower']

    vol_confirm = df['vol_energy'] > vol_normal

    # Combined conditions
    cond_long = alma_bullish & flux_bullish & qs_bullish & vol_confirm
    cond_short = alma_bearish & flux_bearish & qs_bearish & vol_confirm

    # Lookback confirmation
    cond_long = cond_long & cond_long.shift(1).rolling(entry_lookback).sum() >= entry_lookback - 1
    cond_short = cond_short & cond_short.shift(1).rolling(entry_lookback).sum() >= entry_lookback - 1

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if use_vol_filter and pd.isna(df['vol_energy'].iloc[i]):
            continue
        if pd.isna(df['alma'].iloc[i]) or pd.isna(df['flux'].iloc[i]):
            continue

        if cond_long.iloc[i]:
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
        elif cond_short.iloc[i]:
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