import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.
    """
    results = []
    trade_num = 1

    high = df['high']
    low = df['low']
    close = df['close']
    open_ = df['open']
    volume = df['volume']
    time = df['time']

    n = len(df)

    # Detect OB conditions
    def is_up(idx):
        return close.iloc[idx] > open_.iloc[idx]

    def is_down(idx):
        return close.iloc[idx] < open_.iloc[idx]

    def is_ob_up(idx):
        return (is_down(idx + 1) and is_up(idx) and 
                close.iloc[idx] > high.iloc[idx + 1])

    def is_ob_down(idx):
        return (is_up(idx + 1) and is_down(idx) and 
                close.iloc[idx] < low.iloc[idx + 1])

    def is_fvg_up(idx):
        return low.iloc[idx] > high.iloc[idx + 2]

    def is_fvg_down(idx):
        return high.iloc[idx] < low.iloc[idx + 2]

    # Stacked OB + FVG conditions
    ob_up = is_ob_up(1)
    ob_down = is_ob_down(1)
    fvg_up = is_fvg_up(0)
    fvg_down = is_fvg_down(0)

    stacked_bull = ob_up and fvg_up
    stacked_bear = ob_down and fvg_down

    # Wilder RSI implementation
    def wilders_rsi(series, period):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    # Wilder ATR implementation
    def wilders_atr(high_arr, low_arr, close_arr, period):
        tr = pd.DataFrame({
            'tr1': high_arr - low_arr,
            'tr2': (high_arr - close_arr.shift(1)).abs(),
            'tr3': (low_arr - close_arr.shift(1)).abs()
        }).max(axis=1)
        atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        return atr

    # Indicators
    vol_sma9 = volume.rolling(9).mean()
    atr20 = wilders_atr(high, low, close, 20) / 1.5
    loc = close.rolling(54).mean()
    loc_prev = loc.shift(1)
    loc2 = loc > loc_prev

    # FVG conditions
    bfvg = (low > high.shift(2)) 
    sfvg = (high < low.shift(2))
    vol_filt = volume.shift(1) > vol_sma9 * 1.5
    atr_filt = ((low - high.shift(2) > atr20) | (low.shift(2) - high > atr20))
    loc_filt_bull = loc2
    loc_filt_bear = ~loc2

    bfvg = bfvg & vol_filt & atr_filt & loc_filt_bull
    sfvg = sfvg & vol_filt & atr_filt & loc_filt_bear

    # Imbalance conditions
    top_imb_bway = (low.shift(2) <= open_.shift(1)) & (high >= close.shift(1)) & (close < low.shift(1))
    top_imb_xbway = (low.shift(2) <= open_.shift(1)) & (high >= close.shift(1)) & (close > low.shift(1))

    # Time window check (London time: 7:45-9:45 and 14:45-16:45)
    def in_london_window(ts):
        try:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            hour = dt.hour
            minute = dt.minute
            # Morning window: 7:45 to 9:45
            if (hour == 7 and minute >= 45) or (hour == 8) or (hour == 9 and minute <= 45):
                return True
            # Afternoon window: 14:45 to 16:45
            if (hour == 14 and minute >= 45) or (hour == 15) or (hour == 16 and minute <= 45):
                return True
            return False
        except:
            return False

    # PDHL sweep conditions (High/Low sweep)
    prev_day_high = high.shift(1).copy()
    prev_day_low = low.shift(1).copy()

    sweep_high = (close > prev_day_high) & (prev_day_high > open_)
    sweep_low = (close < prev_day_low) & (prev_day_low < open_)

    # Entry conditions
    bull_entry = stacked_bull | (bfvg & (top_imb_bway | top_imb_xbway))
    bear_entry = stacked_bear | (sfvg & (top_imb_bway | top_imb_xbway))

    # Additional: sweep + FVG combo
    bull_sweep_fvg = sweep_high & bfvg
    bear_sweep_fvg = sweep_low & sfvg

    for i in range(20, n):
        ts = int(time.iloc[i])
        if not in_london_window(ts):
            continue

        in_window = True
        if in_window:
            bull_cond = bull_entry.iloc[i] if i < len(bull_entry) else False
            bear_cond = bear_entry.iloc[i] if i < len(bear_entry) else False
            sweep_bull = bull_sweep_fvg.iloc[i] if i < len(bull_sweep_fvg) else False
            sweep_bear = bear_sweep_fvg.iloc[i] if i < len(bear_sweep_fvg) else False

            if bull_cond or sweep_bull:
                results.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(close.iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(close.iloc[i]),
                    'raw_price_b': float(close.iloc[i])
                })
                trade_num += 1

            if bear_cond or sweep_bear:
                results.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(close.iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(close.iloc[i]),
                    'raw_price_b': float(close.iloc[i])
                })
                trade_num += 1

    return results