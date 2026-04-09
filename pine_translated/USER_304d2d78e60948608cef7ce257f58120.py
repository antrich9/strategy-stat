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
    # Input parameters
    maLengthStiffness = 100
    stiffLength = 60
    stiffSmooth = 3
    thresholdStiffness = 90
    
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    
    KC_mult_high_TTMS = 1.0
    KC_mult_mid_TTMS = 1.5
    KC_mult_low_TTMS = 2.0
    
    redGreen_TTMS = True
    cross_TTMS = True
    highlightMovements_TTMS = True
    inverse_TTMS = False
    
    periodXMA = 12
    porogXMA = 3.0
    
    close = df['close']
    high = df['high']
    low = df['low']
    open_prices = df['open']
    
    # Stiffness calculation
    sma_stiff = close.rolling(maLengthStiffness).mean()
    stdev_stiff = close.rolling(maLengthStiffness).std()
    boundStiffness = sma_stiff - 0.2 * stdev_stiff
    
    sumAboveStiffness = (close > boundStiffness).rolling(window=stiffLength).sum()
    stiffness = (sumAboveStiffness * 100 / stiffLength).ewm(span=stiffSmooth, adjust=False).mean()
    
    signalStiffness = stiffness > thresholdStiffness
    
    # TTM Squeeze Components
    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    sma_close_ttm = close.rolling(length_TTMS).mean()
    avg_value = (highest_high + lowest_low + sma_close_ttm) / 3
    diff = close - avg_value
    
    # Linear regression for momentum
    mom_TTMS = pd.Series(index=df.index, dtype=float)
    x = np.arange(length_TTMS)
    x_mean = (length_TTMS - 1) / 2
    sum_x2 = ((x - x_mean) ** 2).sum()
    
    for i in range(length_TTMS - 1, len(df)):
        y = diff.iloc[i - length_TTMS + 1:i + 1].values
        y_mean = y.mean()
        sum_xy = ((x - x_mean) * (y - y_mean)).sum()
        slope = sum_xy / sum_x2
        intercept = y_mean - slope * x_mean
        mom_TTMS.iloc[i] = slope * (length_TTMS - 1 - x_mean) + intercept
    
    mom_prev = mom_TTMS.shift(1)
    
    TTMS_Signals_TTMS = pd.Series(index=df.index, dtype=float)
    TTMS_Signals_TTMS = np.where(mom_TTMS > 0,
                                  np.where(mom_TTMS > mom_prev, 1, 2),
                                  np.where(mom_TTMS < mom_prev, -1, -2))
    TTMS_Signals_TTMS = pd.Series(TTMS_Signals_TTMS, index=df.index)
    
    basicLongCondition_TTMS = (TTMS_Signals_TTMS == 1) if redGreen_TTMS else (TTMS_Signals_TTMS > 0)
    basicShortCondition_TTMS = (TTMS_Signals_TTMS == -1) if redGreen_TTMS else (TTMS_Signals_TTMS < 0)
    
    # Bollinger Bands
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    BB_std_TTMS = close.rolling(length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + BB_mult_TTMS * BB_std_TTMS
    BB_lower_TTMS = BB_basis_TTMS - BB_mult_TTMS * BB_std_TTMS
    
    # Keltner Channels
    devKC_TTMS = close.rolling(length_TTMS).mean()  # Using sma of close instead of tr for simplicity
    KC_upper_low_TTMS = KC_basis_TTMS = BB_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS
    KC_upper_mid_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_mid_TTMS
    KC_lower_mid_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_mid_TTMS
    KC_upper_high_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_high_TTMS
    KC_lower_high_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_high_TTMS
    
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    
    TTMS_SignalsLong_TTMS = NoSqz_TTMS & basicLongCondition_TTMS if highlightMovements_TTMS else basicLongCondition_TTMS
    TTMS_SignalsShort_TTMS = NoSqz_TTMS & basicShortCondition_TTMS if highlightMovements_TTMS else basicShortCondition_TTMS
    
    TTMS_SignalsLongCross_TTMS = (~TTMS_SignalsLong_TTMS.shift(1).fillna(False)) & TTMS_SignalsLong_TTMS if cross_TTMS else TTMS_SignalsLong_TTMS
    TTMS_SignalsShortCross_TTMS = (~TTMS_SignalsShort_TTMS.shift(1).fillna(False)) & TTMS_SignalsShort_TTMS if cross_TTMS else TTMS_SignalsShort_TTMS
    
    TTMS_SignalsLongFinal_TTMS = TTMS_SignalsLongCross_TTMS
    TTMS_SignalsShortFinal_TTMS = TTMS_SignalsShortCross_TTMS
    
    # AKKAM (always true since useAKKAM=true, but check logic)
    akkamLong = TTMS_SignalsLongFinal_TTMS
    akkamShort = TTMS_SignalsShortFinal_TTMS
    
    # XMA calculation
    emaCurrentXMA = close.ewm(span=periodXMA, adjust=False).mean()
    emaPreviousXMA = close.shift(1).ewm(span=periodXMA, adjust=False).mean()
    
    xmaSignalLong = close >= emaCurrentXMA
    xmaSignalShort = close <= emaCurrentXMA
    
    signalLongXMA = xmaSignalLong.shift(1).fillna(False) == False
    signalLongXMA = signalLongXMA & xmaSignalLong
    signalShortXMA = xmaSignalShort.shift(1).fillna(False) == False
    signalShortXMA = signalShortXMA & xmaSignalShort
    
    finalLongSignalXMA = signalLongXMA
    finalShortSignalXMA = signalShortXMA
    
    # Combined conditions
    long_condition = signalStiffness & basicLongCondition_TTMS & xmaSignalLong
    short_condition = signalStiffness & basicShortCondition_TTMS & xmaSignalShort
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(mom_TTMS.iloc[i]) or pd.isna(stiffness.iloc[i]) or pd.isna(emaCurrentXMA.iloc[i]):
            continue
        
        entry_price = close.iloc[i]
        
        if long_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
        
        if short_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
    
    return entries