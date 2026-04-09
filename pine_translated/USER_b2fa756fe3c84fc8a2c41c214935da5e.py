import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    lookback = 3
    pivotType = "High/Low"
    oneSet = 'D'
    twoSet = '240'
    threeSet = '60'
    
    tf_multipliers = {'D': 1440, '240': 240, '60': 60}
    chart_tf = 1
    
    def get_htf_data(df_input, tf_set, tf_name):
        multiplier = tf_multipliers.get(tf_set, tf_multipliers.get(tf_set.replace('M', '43200').replace('W', '10080'), 1))
        period = max(1, multiplier // chart_tf)
        
        if period <= 1:
            return (df_input['close'].copy(), df_input['open'].copy(), 
                    df_input['high'].copy(), df_input['low'].copy())
        
        htf_close = df_input['close'].iloc[::period].reindex(df_input.index).ffill()
        htf_open = df_input['open'].iloc[::period].reindex(df_input.index).ffill()
        htf_high = df_input['high'].iloc[::period].reindex(df_input.index).ffill()
        htf_low = df_input['low'].iloc[::period].reindex(df_input.index).ffill()
        
        return htf_close, htf_open, htf_high, htf_low
    
    htf_close_1, htf_open_1, htf_high_1, htf_low_1 = get_htf_data(df, oneSet, '1')
    htf_close_2, htf_open_2, htf_high_2, htf_low_2 = get_htf_data(df, twoSet, '2')
    htf_close_3, htf_open_3, htf_high_3, htf_low_3 = get_htf_data(df, threeSet, '3')
    
    bgColor_1 = (htf_close_1 > htf_open_1).astype(int).replace(0, -1)
    bgColor_2 = (htf_close_2 > htf_open_2).astype(int).replace(0, -1)
    bgColor_3 = (htf_close_3 > htf_open_3).astype(int).replace(0, -1)
    
    prev_trend_1 = bgColor_1
    prev_trend_2 = bgColor_2
    prev_trend_3 = bgColor_3
    
    def get_pivots(df_input, pivot_type, lookback_val):
        if pivot_type == "Close":
            src = df_input['close']
        else:
            src_high = df_input['high']
            src_low = df_input['low']
        
        if pivot_type == "Close":
            window_high = src.rolling(lookback_val*2+1, center=True).max()
            ph = (src == window_high) & (src.shift(lookback_val) == window_high)
            window_low = src.rolling(lookback_val*2+1, center=True).min()
            pl = (src == window_low) & (src.shift(lookback_val) == window_low)
            highLevel = src.shift(lookback_val).where(ph)
            lowLevel = src.shift(lookback_val).where(pl)
        else:
            window_high = src_high.rolling(lookback_val*2+1, center=True).max()
            ph = (src_high == window_high) & (src_high.shift(lookback_val) == window_high)
            window_low = src_low.rolling(lookback_val*2+1, center=True).min()
            pl = (src_low == window_low) & (src_low.shift(lookback_val) == window_low)
            highLevel = src_high.shift(lookback_val).where(ph)
            lowLevel = src_low.shift(lookback_val).where(pl)
        
        barsSinceHigh = pd.Series(np.nan, index=df_input.index)
        barsSinceLow = pd.Series(np.nan, index=df_input.index)
        
        for i in range(lookback_val, len(df_input)):
            if ph.iloc[i]:
                barsSinceHigh.iloc[i] = 0
            elif i > 0 and not np.isnan(barsSinceHigh.iloc[i-1]):
                barsSinceHigh.iloc[i] = barsSinceHigh.iloc[i-1] + 1
        
        for i in range(lookback_val, len(df_input)):
            if pl.iloc[i]:
                barsSinceLow.iloc[i] = 0
            elif i > 0 and not np.isnan(barsSinceLow.iloc[i-1]):
                barsSinceLow.iloc[i] = barsSinceLow.iloc[i-1] + 1
        
        barsSinceHigh = barsSinceHigh + lookback_val
        barsSinceLow = barsSinceLow + lookback_val
        
        timeSinceHigh = df_input['time'].shift(barsSinceHigh.astype(Int64))
        timeSinceLow = df_input['time'].shift(barsSinceLow.astype(Int64))
        
        return ph, pl, highLevel, lowLevel, barsSinceHigh, barsSinceLow, timeSinceHigh, timeSinceLow
    
    ph_01, pl_01, hL_01, lL_01, bsSH_01, bsSL_01, tSH_01, tSL_01 = get_pivots(df, pivotType, lookback)
    ph_02, pl_02, hL_02, lL_02, bsSH_02, bsSL_02, tSH_02, tSL_02 = get_pivots(df, pivotType, lookback)
    ph_03, pl_03, hL_03, lL_03, bsSH_03, bsSL_03, tSH_03, tSL_03 = get_pivots(df, pivotType, lookback)
    
    def f_signal(highLevel, lowLevel, df_input):
        uptrendSignal = df_input['high'] > highLevel
        downtrendSignal = df_input['low'] < lowLevel
        
        inUptrend = pd.Series(False, index=df_input.index)
        inDowntrend = pd.Series(False, index=df_input.index)
        
        for i in range(1, len(df_input)):
            if pd.notna(uptrendSignal.iloc[i]) and uptrendSignal.iloc[i]:
                inUptrend.iloc[i] = True
            elif pd.notna(downtrendSignal.iloc[i]) and downtrendSignal.iloc[i]:
                inUptrend.iloc[i] = False
            else:
                inUptrend.iloc[i] = inUptrend.iloc[i-1]
        
        inDowntrend = ~inUptrend
        return uptrendSignal, downtrendSignal, inUptrend, inDowntrend
    
    uptrendSignal1, downtrendSignal1, inUptrend1, inDowntrend1 = f_signal(hL_01, lL_01, df)
    uptrendSignal2, downtrendSignal2, inUptrend2, inDowntrend2 = f_signal(hL_02, lL_02, df)
    uptrendSignal3, downtrendSignal3, inUptrend3, inDowntrend3 = f_signal(hL_03, lL_03, df)
    
    combined_uptrend = inUptrend1 & inUptrend2 & inUptrend3
    combined_downtrend = inDowntrend1 & inDowntrend2 & inDowntrend3
    
    entries = []
    trade_num = 1
    
    inUptrend_prev = False
    inDowntrend_prev = False
    
    for i in range(1, len(df)):
        bull_count_1 = 1 if htf_close_1.iloc[i] > htf_open_1.iloc[i] else 0
        bull_count_2 = 1 if htf_close_2.iloc[i] > htf_open_2.iloc[i] else 0
        bull_count_3 = 1 if htf_close_3.iloc[i] > htf_open_3.iloc[i] else 0
        bull_total = bull_count_1 + bull_count_2 + bull_count_3
        
        bear_count_1 = 1 if htf_close_1.iloc[i] <= htf_open_1.iloc[i] else 0
        bear_count_2 = 1 if htf_close_2.iloc[i] <= htf_open_2.iloc[i] else 0
        bear_count_3 = 1 if htf_close_3.iloc[i] <= htf_open_3.iloc[i] else 0
        bear_total = bear_count_1 + bear_count_2 + bear_count_3
        
        inUptrend_curr = combined_uptrend.iloc[i] and inUptrend1.iloc[i] and inUptrend2.iloc[i] and inUptrend3.iloc[i]
        inDowntrend_curr = combined_downtrend.iloc[i] and inDowntrend1.iloc[i] and inDowntrend2.iloc[i] and inDowntrend3.iloc[i]
        
        long_entry = (inUptrend_curr and not inUptrend_prev) or (inUptrend1.iloc[i] and inUptrend2.iloc[i] and inUptrend3.iloc[i] and bull_total >= 2)
        short_entry = (inDowntrend_curr and not inDowntrend_prev) or (inDowntrend1.iloc[i] and inDowntrend2.iloc[i] and inDowntrend3.iloc[i] and bear_total >= 2)
        
        direction = None
        if long_entry:
            direction = 'long'
        elif short_entry:
            direction = 'short'
        
        if direction:
            ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        
        inUptrend_prev = inUptrend_curr
        inDowntrend_prev = inDowntrend_curr
    
    return entries