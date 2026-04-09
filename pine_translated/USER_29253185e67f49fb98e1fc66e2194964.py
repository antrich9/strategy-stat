import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    open_price = df['open']
    high = df['high']
    low = df['low']

    ema8 = close.ewm(span=8, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    body = (close - open_price).abs()
    rng = high - low
    dojiPerc = 0.30
    isDoji = (rng > 0) & ((body / rng) <= dojiPerc)

    emaBull = (ema8 > ema20) & (ema20 > ema50)
    emaBear = (ema8 < ema20) & (ema20 < ema50)

    closeInUpper33 = (close - low) / rng >= 0.67
    closeInLower33 = (high - close) / rng >= 0.67
    bodyInUpper33 = (open_price >= low + rng * 0.67) & (close >= low + rng * 0.67)
    bodyInLower33 = (open_price <= low + rng * 0.33) & (close <= low + rng * 0.33)

    bullSignal = emaBull & isDoji & (low <= ema8) & closeInUpper33 & bodyInUpper33
    bearSignal = emaBear & isDoji & (high >= ema8) & closeInLower33 & bodyInLower33

    bullOnly = bullSignal & ~bearSignal
    bearOnly = bearSignal & ~bullSignal

    offset = 0.0004
    entries = []
    trade_num = 1
    pending_signals = []

    for i in range(len(df)):
        ts = int(df['time'].iloc[i])
        entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        current_bar = i

        if bullOnly.iloc[i]:
            entry_p = high.iloc[i] + offset
            pending_signals.append({'sig_bar': i, 'entry_price': entry_p, 'direction': 'long'})
        elif bearOnly.iloc[i]:
            entry_p = low.iloc[i] - offset
            pending_signals.append({'sig_bar': i, 'entry_price': entry_p, 'direction': 'short'})

        signals_to_remove = []
        for sig in pending_signals:
            sig_bar = sig['sig_bar']
            if current_bar == sig_bar + 1:
                if sig['direction'] == 'long' and high.iloc[i] >= sig['entry_price']:
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': ts,
                        'entry_time': entry_time_str,
                        'entry_price_guess': sig['entry_price'],
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': sig['entry_price'],
                        'raw_price_b': sig['entry_price']
                    })
                    trade_num += 1
                    signals_to_remove.append(sig)
                elif sig['direction'] == 'short' and low.iloc[i] <= sig['entry_price']:
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': ts,
                        'entry_time': entry_time_str,
                        'entry_price_guess': sig['entry_price'],
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': sig['entry_price'],
                        'raw_price_b': sig['entry_price']
                    })
                    trade_num += 1
                    signals_to_remove.append(sig)
                else:
                    signals_to_remove.append(sig)
            elif current_bar > sig_bar + 1:
                signals_to_remove.append(sig)

        for sig in signals_to_remove:
            pending_signals.remove(sig)

    return entries