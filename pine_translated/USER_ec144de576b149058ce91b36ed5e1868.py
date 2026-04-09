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
    # Strategy constants (matching Pine Script inputs)
    length_tema = 9
    length_ema = 9
    rsi_period = 11
    band_length = 31
    lengthrsipl = 1
    lengthtradesl = 9

    close = df['close']

    # Triple EMA (TEMA)
    ema1 = close.ewm(span=length_tema, adjust=False).mean()
    ema2 = ema1.ewm(span=length_tema, adjust=False).mean()
    ema3 = ema2.ewm(span=length_tema, adjust=False).mean()
    tema = 3 * (ema1 - ema2) + ema3

    # EMA of close
    ema = close.ewm(span=length_ema, adjust=False).mean()

    # Wilder RSI (ta.rsi)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # TDI bands
    ma = rsi.rolling(band_length).mean()
    std = rsi.rolling(band_length).std(ddof=0)      # population stdev (default in Pine)
    offs = 1.6185 * std
    up = ma + offs
    dn = ma - offs
    mid = (up + dn) / 2

    # Fast & slow SMA of RSI (fastMA used in signal logic)
    fastMA = rsi.rolling(lengthrsipl).mean()        # length = 1 → same as rsi
    slowMA = rsi.rolling(lengthtradesl).mean()     # not required for entry

    # Background condition: close > ema AND close > tema
    bg_condition = (close > ema) & (close > tema)

    # Stateful variables for signal generation
    last_break_below_lower_band = False
    prev_bg_condition = False
    signal = pd.Series(False, index=df.index)

    # Loop over bars to compute signal (respecting NaN values)
    for i in range(len(df)):
        # Skip bars with missing required data
        if (pd.isna(close.iloc[i]) or pd.isna(ema.iloc[i]) or pd.isna(tema.iloc[i]) or
            pd.isna(rsi.iloc[i]) or pd.isna(up.iloc[i]) or pd.isna(dn.iloc[i]) or
            pd.isna(fastMA.iloc[i])):
            continue

        # Update lastBreakBelowLowerBand
        if fastMA.iloc[i] > up.iloc[i]:
            last_break_below_lower_band = False
        if fastMA.iloc[i] < dn.iloc[i]:
            last_break_below_lower_band = True

        bg = bg_condition.iloc[i]

        # Signal = bgCondition AND NOT prevBgCondition AND lastBreakBelowLowerBand
        if bg and not prev_bg_condition and last_break_below_lower_band:
            signal.iloc[i] = True

        # Update previous bg condition for next bar
        prev_bg_condition = bg

    # Build entry list
    entries = []
    trade_num = 1
    for i in df.index:
        if signal.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries