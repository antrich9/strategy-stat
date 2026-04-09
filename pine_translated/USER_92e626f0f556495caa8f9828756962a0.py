import pandas as pd
import numpy as np
from datetime import datetime, timezone

def alma_filter(source, window, offset, sigma):
    m = (offset - 1) * np.log(2)
    w = np.exp(-np.arange(window) ** 2 / (2 * sigma ** 2))
    w = w / w.sum()
    padded = np.concatenate([np.full(window - 1, np.nan), source.values])
    result = np.full(len(source), np.nan)
    for i in range(window - 1, len(padded)):
        result[i - (window - 1)] = np.sum(padded[i - window + 1:i + 1] * w[::-1])
    return pd.Series(result, index=source.index)

def fluxwave_oscillator(source, smooth, length, alma_offset, alma_sigma):
    flux = source.ewm(span=smooth, adjust=False).mean()
    flux_alma = alma_filter(flux, length, alma_offset, alma_sigma)
    wave1 = flux_alma.ewm(span=5, adjust=False).mean()
    wave2 = flux_alma.ewm(span=12, adjust=False).mean()
    wave3 = flux_alma.ewm(span=26, adjust=False).mean()
    fluxline = wave1 - wave2 - wave3
    flux_band_mult = 1.0
    upper = flux_alma + flux_alma.rolling(length).std() * flux_band_mult
    lower = flux_alma - flux_alma.rolling(length).std() * flux_band_mult
    return fluxline, flux_alma, upper, lower

def quicksilver_bands(source, period, deviation):
    mid = source.rolling(period).mean()
    std = source.rolling(period).std()
    upper = mid + std * deviation
    lower = mid - std * deviation
    return mid, upper, lower

def volume_energy(volume, length, high_thresh, normal_thresh, low_thresh):
    normalized = (volume - volume.rolling(length).min()) / (volume.rolling(length).max() - volume.rolling(length).min() + 1e-10)
    normalized *= 100
    vol_energy = normalized.ewm(span=length, adjust=False).mean()
    high_energy = vol_energy > high_thresh
    normal_energy = vol_energy > normal_thresh
    low_energy = vol_energy > low_thresh
    return vol_energy, high_energy, normal_energy, low_energy

def wilder_rsi(source, length):
    delta = source.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def wilder_atr(high, low, close, length):
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

def network_activity_metric(close, period, threshold):
    returns = close.pct_change()
    volatility = returns.rolling(period).std()
    growth = close.rolling(period).apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) if len(x) > 0 else 0)
    activity_score = growth / (volatility + 1e-10)
    return activity_score > threshold

def holder_behavior_metric(close, volume, period, accum_threshold):
    price_ma = close.rolling(period).mean()
    vol_ma = volume.rolling(period).mean()
    accumulation = (close - price_ma) / (price_ma + 1e-10)
    vol_ratio = volume / (vol_ma + 1e-10)
    holder_score = (accumulation * 0.6 + vol_ratio * 0.4) * 100
    return holder_score > accum_threshold

def exchange_flow_metric(close, volume, period, outflow_threshold):
    volatility = close.rolling(period).std()
    vol_normalized = volume / (volume.rolling(period).mean() + 1e-10)
    flow_ratio = vol_normalized / (volatility + 1e-10)
    return flow_ratio > outflow_threshold

def whale_activity_detection(close, volume, period, threshold_mult):
    returns = close.pct_change()
    vol_zscore = (volume - volume.rolling(period).mean()) / (volume.rolling(period).std() + 1e-10)
    whale_signal = vol_zscore > threshold_mult
    price_momentum = returns.rolling(5).sum()
    whale_activity = whale_signal & (np.abs(price_momentum) > 0.02)
    return whale_activity

def mvrv_analysis(close, period, overbought_level, oversold_level):
    realized = close.rolling(period).mean()
    mvrv = close / (realized + 1e-10)
    return mvrv

def generate_entries(df: pd.DataFrame) -> list:
    source_alma = df['close']
    alma_window = 9
    alma_offset = 0.85
    alma_sigma = 6.0
    flux_smooth = 1
    flux_length = 50
    flux_offset = 0.85
    flux_sigma = 6
    flux_band_mult = 1.0
    qs_period = 20
    qs_deviation = 2.0
    vol_length = 50
    vol_high = 150
    vol_normal = 75
    vol_low = 75
    entry_mode = "Combined"
    use_alma_signal = True
    use_flux_signal = True
    use_qs_signal = True
    use_vol_filter = True
    use_onchain_filter = True
    entry_lookback = 5
    entry_threshold = 0.5
    qfn_cooldown = 3
    onchain_enabled = True
    onchain_network_period = 14
    onchain_network_threshold = 1.2
    onchain_holder_period = 30
    onchain_accumulation_threshold = 60.0
    onchain_exchange_period = 7
    onchain_outflow_threshold = 1.5
    onchain_whale_filter = True
    onchain_whale_threshold = 2.0
    onchain_mvrv_filter = True
    onchain_mvrv_overbought = 3.5
    onchain_mvrv_oversold = 1.0

    alma_line = alma_filter(source_alma, alma_window, alma_offset, alma_sigma)
    flux_line, flux_alma, flux_upper, flux_lower = fluxwave_oscillator(df['close'], flux_smooth, flux_length, flux_offset, flux_sigma)
    qs_mid, qs_upper, qs_lower = quicksilver_bands(df['close'], qs_period, qs_deviation)
    vol_energy, vol_high_energy, vol_normal_energy, vol_low_energy = volume_energy(df['volume'], vol_length, vol_high, vol_normal, vol_low)
    flux_signal_line = flux_line.ewm(span=9, adjust=False).mean()
    flux_rsi = wilder_rsi(flux_line, 14)
    flux_momentum = flux_line.diff(5)

    network_growth = network_activity_metric(df['close'], onchain_network_period, onchain_network_threshold)
    holder_accumulation = holder_behavior_metric(df['close'], df['volume'], onchain_holder_period, onchain_accumulation_threshold)
    exchange_outflow = exchange_flow_metric(df['close'], df['volume'], onchain_exchange_period, onchain_outflow_threshold)
    whale_activity = whale_activity_detection(df['close'], df['volume'], 20, onchain_whale_threshold) if onchain_whale_filter else pd.Series(True, index=df.index)
    mvrv = mvrv_analysis(df['close'], 30, onchain_mvrv_overbought, onchain_mvrv_oversold)
    mvrv_safe = mvrv.fillna(1.0)
    mvrv_not_extreme = (mvrv_safe < onchain_mvrv_overbought) & (mvrv_safe > onchain_mvrv_oversold)

    alma_bullish_cross = (source_alma > alma_line) & (source_alma.shift(1) <= alma_line.shift(1))
    alma_bearish_cross = (source_alma < alma_line) & (source_alma.shift(1) >= alma_line.shift(1))

    flux_bullish = (flux_signal_line > flux_signal_line.shift(1)) & (flux_rsi < 70)
    flux_bearish = (flux_signal_line < flux_signal_line.shift(1)) & (flux_rsi > 30)
    flux_os = flux_rsi < 30
    flux_ob = flux_rsi > 70

    qs_bullish_breakout = df['close'] > qs_upper
    qs_bearish_breakout = df['close'] < qs_lower

    vol_pass = vol_normal_energy if use_vol_filter else pd.Series(True, index=df.index)
    onchain_pass = network_growth & holder_accumulation & exchange_outflow & whale_activity & mvrv_not_extreme if use_onchain_filter else pd.Series(True, index=df.index)

    cooldown = qfn_cooldown

    entries = []
    trade_num = 1
    last_trade_bar = -9999

    for i in range(1, len(df)):
        if i - last_trade_bar <= cooldown:
            continue

        if alma_line.iloc[i] is np.nan or flux_line.iloc[i] is np.nan:
            continue

        alma_cond = alma_bullish_cross.iloc[i] if use_alma_signal else False
        flux_cond = flux_bullish.iloc[i] if use_flux_signal else False
        qs_cond = qs_bullish_breakout.iloc[i] if use_qs_signal else False

        if entry_mode == "ALMA":
            alma_entry = alma_bullish_cross.iloc[i] and vol_pass.iloc[i] and onchain_pass.iloc[i] and use_alma_signal
        elif entry_mode == "FluxWave":
            alma_entry = flux_bullish.iloc[i] and vol_pass.iloc[i] and onchain_pass.iloc[i] and use_flux_signal
        elif entry_mode == "QuickSilver":
            alma_entry = qs_bullish_breakout.iloc[i] and vol_pass.iloc[i] and onchain_pass.iloc[i] and use_qs_signal
        elif entry_mode == "Custom":
            alma_entry = (alma_cond or flux_cond or qs_cond) and vol_pass.iloc[i] and onchain_pass.iloc[i]
        else:
            alma_entry = alma_cond and flux_cond and qs_cond and vol_pass.iloc[i] and onchain_pass.iloc[i]

        if alma_entry:
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
            last_trade_bar = i

    return entries