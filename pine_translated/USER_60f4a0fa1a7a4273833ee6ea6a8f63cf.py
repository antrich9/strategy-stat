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
    # === CONSTANTS ===
    lookback = 30
    eqTolAtrMult = 0.15
    minTouches = 2
    atrLen = 14
    minBodyPct = 0.60
    minRangeAtr = 0.80
    useFVG = True
    fvgWaitBars = 10
    fvgMinTicks = 3
    useEma200 = True
    emaLen = 200
    useSession = False
    sym_min_tick = 1.0

    # === ATR (Wilder) ===
    high = df['high']
    low = df['low']
    close = df['close']
    open_arr = df['open']

    tr = np.maximum(high.diff().abs(), np.maximum(
        (high - low.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ))
    atr = tr.ewm(span=atrLen, adjust=False).mean()

    # === EMA200 ===
    ema200 = close.ewm(span=emaLen, adjust=False).mean()

    # === Candle Properties ===
    rng = high - low
    body = (close - open_arr).abs()
    body_pct = np.where(rng > 0, body / rng, 0.0)

    # === Displacement Candle ===
    is_displacement_bull = (close > open_arr) & (body_pct >= minBodyPct) & (rng >= atr * minRangeAtr)
    is_displacement_bear = (close < open_arr) & (body_pct >= minBodyPct) & (rng >= atr * minRangeAtr)

    # === Trend Filter ===
    trend_long_ok = (~useEma200) | (close >= ema200)
    trend_short_ok = (~useEma200) | (close <= ema200)

    # === FVG Detection ===
    bullish_fvg = low > high.shift(2)
    bullish_fvg_size = np.where(bullish_fvg, low - high.shift(2), 0.0)
    bullish_fvg_valid = bullish_fvg & (bullish_fvg_size / sym_min_tick >= fvgMinTicks)

    bearish_fvg = high < low.shift(2)
    bearish_fvg_size = np.where(bearish_fvg, low.shift(2) - high, 0.0)
    bearish_fvg_valid = bearish_fvg & (bearish_fvg_size / sym_min_tick >= fvgMinTicks)

    # === Liquidity Detection ===
    liq_high_level = high.shift(1)
    liq_low_level = low.shift(1)

    tol = atr * eqTolAtrMult

    def count_near(series, level):
        diff = (series - level).abs()
        threshold = atr * eqTolAtrMult
        return (diff <= threshold).rolling(window=lookback).sum()

    high_touches = count_near(high, liq_high_level)
    low_touches = count_near(low, liq_low_level)

    has_buy_side_liq = high_touches >= minTouches
    has_sell_side_liq = low_touches >= minTouches

    # === Sweep Detection ===
    sweep_sell_side = has_sell_side_liq & (low < liq_low_level - tol) & (close > liq_low_level)
    sweep_buy_side = has_buy_side_liq & (high > liq_high_level + tol) & (close < liq_high_level)

    # === Entry Logic ===
    entries = []
    trade_num = 1

    last_sell_sweep_bar = pd.Series(np.nan, index=df.index)
    last_buy_sweep_bar = pd.Series(np.nan, index=df.index)

    for i in df.index:
        if i < 2:
            continue

        # Track sweep bars
        if sweep_sell_side.iloc[i]:
            last_sell_sweep_bar.iloc[i] = i
        else:
            last_sell_sweep_bar.iloc[i] = last_sell_sweep_bar.iloc[i-1] if i > 0 else np.nan

        if sweep_buy_side.iloc[i]:
            last_buy_sweep_bar.iloc[i] = i
        else:
            last_buy_sweep_bar.iloc[i] = last_buy_sweep_bar.iloc[i-1] if i > 0 else np.nan

        bars_since_sell = i - last_sell_sweep_bar.iloc[i] if not pd.isna(last_sell_sweep_bar.iloc[i]) else 999
        bars_since_buy = i - last_buy_sweep_bar.iloc[i] if not pd.isna(last_buy_sweep_bar.iloc[i]) else 999

        if bars_since_sell > fvgWaitBars:
            last_sell_sweep_bar.iloc[i] = np.nan
        if bars_since_buy > fvgWaitBars:
            last_buy_sweep_bar.iloc[i] = np.nan

        ts = int(df['time'].iloc[i])
        price = float(df['close'].iloc[i])

        if useFVG:
            # LONG ENTRY
            long_setup = sweep_sell_side.iloc[i] or (bars_since_sell > 0 and bars_since_sell <= fvgWaitBars)
            long_entry = long_setup and bullish_fvg_valid.iloc[i] and trend_long_ok.iloc[i]

            # SHORT ENTRY
            short_setup = sweep_buy_side.iloc[i] or (bars_since_buy > 0 and bars_since_buy <= fvgWaitBars)
            short_entry = short_setup and bearish_fvg_valid.iloc[i] and trend_short_ok.iloc[i]
        else:
            bull_sweep_disp = sweep_sell_side.iloc[i] and is_displacement_bull.iloc[i] and trend_long_ok.iloc[i]
            bear_sweep_disp = sweep_buy_side.iloc[i] and is_displacement_bear.iloc[i] and trend_short_ok.iloc[i]
            long_entry = bull_sweep_disp
            short_entry = bear_sweep_disp

        if long_entry:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
            last_sell_sweep_bar.iloc[i] = np.nan

        if short_entry:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
            last_buy_sweep_bar.iloc[i] = np.nan

    return entries