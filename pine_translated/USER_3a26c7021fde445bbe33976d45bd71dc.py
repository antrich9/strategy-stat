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
    # Input parameters from Pine Script
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    redGreen_TTMS = True
    cross_TTMS = True
    inverse_TTMS = False
    highlightMovements_TTMS = True
    use_TTMS = True

    close = df['close']
    high = df['high']
    low = df['low']

    # Bollinger Bands
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = BB_mult_TTMS * close.rolling(length_TTMS).std()

    # Keltner Channels
    KC_mult_high_TTMS = 1.0
    KC_mult_mid_TTMS = 1.5
    KC_mult_low_TTMS = 2.0
    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    tr = np.maximum(
        np.maximum(high - low, np.abs(high - close.shift(1))),
        np.abs(low - close.shift(1))
    )
    devKC_TTMS = pd.Series(tr).rolling(length_TTMS).mean()

    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS
    KC_upper_high_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_high_TTMS
    KC_lower_high_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_high_TTMS
    KC_upper_mid_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_mid_TTMS
    KC_lower_mid_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_mid_TTMS
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS

    # Squeeze Conditions
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    LowSqz_TTMS = (BB_lower_TTMS >= KC_lower_low_TTMS) | (BB_upper_TTMS <= KC_upper_low_TTMS)

    # Momentum Oscillator (linreg)
    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    avg_val = (highest_high + lowest_low) / 2
    sma_close = close.rolling(length_TTMS).mean()
    mom_source = close - (avg_val + sma_close) / 2

    # Linear regression using numpy polyfit
    def linreg(series, length):
        result = pd.Series(np.nan, index=series.index)
        for i in range(length - 1, len(series)):
            if pd.notna(series.iloc[i]) and not series.iloc[i-length:i].isna().any():
                y = series.iloc[i-length+1:i+1].values
                x = np.arange(length)
                try:
                    coef = np.polyfit(x, y, 1)
                    result.iloc[i] = coef[0] * (length - 1) + coef[1]
                except:
                    pass
        return result

    mom_TTMS = linreg(mom_source, length_TTMS)
    mom_prev = mom_TTMS.shift(1)

    # Heartbeat Logic
    iff_1_TTMS_no = pd.Series(1, index=mom_TTMS.index).where(mom_TTMS > mom_prev, other=2)
    iff_2_TTMS_no = pd.Series(-1, index=mom_TTMS.index).where(mom_TTMS < mom_prev, other=-2)
    TTMS_Signals_TTMS = pd.Series(0, index=mom_TTMS.index).where(mom_TTMS == 0, other=iff_1_TTMS_no.where(mom_TTMS > 0, iff_2_TTMS_no))

    basicLongCondition_TTMS = TTMS_Signals_TTMS.where(redGreen_TTMS, mom_TTMS > 0).replace({True: TTMS_Signals_TTMS == 1, False: TTMS_Signals_TTMS > 0}) if redGreen_TTMS else mom_TTMS > 0
    basicShortCondition_TTMS = TTMS_Signals_TTMS.where(redGreen_TTMS, mom_TTMS < 0).replace({True: TTMS_Signals_TTMS == -1, False: TTMS_Signals_TTMS < 0}) if redGreen_TTMS else mom_TTMS < 0

    if redGreen_TTMS:
        basicLongCondition_TTMS = TTMS_Signals_TTMS == 1
        basicShortCondition_TTMS = TTMS_Signals_TTMS == -1
    else:
        basicLongCondition_TTMS = TTMS_Signals_TTMS > 0
        basicShortCondition_TTMS = TTMS_Signals_TTMS < 0

    TTMS_SignalsLong_TTMS = (NoSqz_TTMS & basicLongCondition_TTMS) if highlightMovements_TTMS else basicLongCondition_TTMS
    TTMS_SignalsShort_TTMS = (NoSqz_TTMS & basicShortCondition_TTMS) if highlightMovements_TTMS else basicShortCondition_TTMS

    TTMS_SignalsLongPrev_TTMS = TTMS_SignalsLong_TTMS.shift(1)
    TTMS_SignalsShortPrev_TTMS = TTMS_SignalsShort_TTMS.shift(1)

    TTMS_SignalsLongCross_TTMS = (~TTMS_SignalsLongPrev_TTMS.fillna(False)) & TTMS_SignalsLong_TTMS if cross_TTMS else TTMS_SignalsLong_TTMS
    TTMS_SignalsShortCross_TTMS = (~TTMS_SignalsShortPrev_TTMS.fillna(False)) & TTMS_SignalsShort_TTMS if cross_TTMS else TTMS_SignalsShort_TTMS

    TTMS_SignalsLongFinal_TTMS = TTMS_SignalsShortCross_TTMS if (use_TTMS and inverse_TTMS) else (TTMS_SignalsLongCross_TTMS if use_TTMS else pd.Series(True, index=df.index))
    TTMS_SignalsShortFinal_TTMS = TTMS_SignalsLongCross_TTMS if (use_TTMS and inverse_TTMS) else (TTMS_SignalsShortCross_TTMS if use_TTMS else pd.Series(True, index=df.index))

    long_condition = basicLongCondition_TTMS
    short_condition = basicShortCondition_TTMS

    entries = []
    trade_num = 1
    in_position = False
    position_direction = None

    for i in range(len(df)):
        if in_position:
            continue

        entry_price = close.iloc[i]
        ts = int(df['time'].iloc[i])

        if long_condition.iloc[i] if pd.notna(long_condition.iloc[i]) else False:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            in_position = True
            position_direction = 'long'
        elif short_condition.iloc[i] if pd.notna(short_condition.iloc[i]) else False:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            in_position = True
            position_direction = 'short'

    return entries