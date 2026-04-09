import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['time_dt'].dt.hour
    
    asian_start = df['hour'] == 0
    london_start = df['hour'] == 8
    ny_start = df['hour'] == 13
    
    asian_session = (df['hour'] >= 0) & (df['hour'] < 9)
    london_session = (df['hour'] >= 8) & (df['hour'] < 17)
    ny_session = (df['hour'] >= 13) & (df['hour'] < 22)
    trading_window = (df['hour'] >= 14) & (df['hour'] < 17)
    
    df['asian_session_start'] = asian_start
    df['london_session_start'] = london_start
    df['ny_session_start'] = ny_start
    
    df['asian_grp'] = df['asian_session_start'].cumsum()
    df['london_grp'] = df['london_session_start'].cumsum()
    df['ny_grp'] = df['ny_session_start'].cumsum()
    
    def sess_max(gp, vals):
        mask = df[gp] > 0
        out = pd.Series(index=df.index, dtype=float)
        out[mask] = df.loc[mask, 'high'].where(df.loc[mask, gp] == 1).ffill()
        grp_max = df.loc[mask].groupby(gp)['high'].cummax()
        out.loc[mask] = grp_max
        return out
    
    def sess_min(gp, vals):
        mask = df[gp] > 0
        out = pd.Series(index=df.index, dtype=float)
        grp_min = df.loc[mask].groupby(gp)['low'].cummin()
        out.loc[mask] = grp_min
        return out
    
    asian_sess = df['asian_session_start'].cumsum()
    london_sess = df['london_session_start'].cumsum()
    ny_sess = df['ny_session_start'].cumsum()
    
    asian_high = pd.Series(index=df.index, dtype=float)
    asian_low = pd.Series(index=df.index, dtype=float)
    asian_open = pd.Series(index=df.index, dtype=float)
    asian_close = pd.Series(index=df.index, dtype=float)
    
    london_high = pd.Series(index=df.index, dtype=float)
    london_low = pd.Series(index=df.index, dtype=float)
    london_open = pd.Series(index=df.index, dtype=float)
    london_close = pd.Series(index=df.index, dtype=float)
    
    ny_open = pd.Series(index=df.index, dtype=float)
    
    a_grp = 0
    l_grp = 0
    n_grp = 0
    a_h = a_l = a_o = a_c = float('nan')
    l_h = l_l = l_o = l_c = float('nan')
    n_o = float('nan')
    
    for i in df.index:
        if df.loc[i, 'asian_session_start']:
            if a_grp > 0:
                if not pd.isna(asian_high.loc[df.loc[i, 'time']] if df.loc[i, 'time'] in asian_high.index else float('nan')):
                    pass
            a_grp = asian_sess.loc[i]
            a_h = df.loc[i, 'high']
            a_l = df.loc[i, 'low']
            a_o = df.loc[i, 'open']
            a_c = df.loc[i, 'close']
        elif df.loc[i, 'asian_session']:
            a_h = max(a_h, df.loc[i, 'high']) if not pd.isna(a_h) else df.loc[i, 'high']
            a_l = min(a_l, df.loc[i, 'low']) if not pd.isna(a_l) else df.loc[i, 'low']
            a_c = df.loc[i, 'close']
        
        if df.loc[i, 'london_session_start']:
            l_grp = london_sess.loc[i]
            l_h = df.loc[i, 'high']
            l_l = df.loc[i, 'low']
            l_o = df.loc[i, 'open']
            l_c = df.loc[i, 'close']
        elif df.loc[i, 'london_session']:
            l_h = max(l_h, df.loc[i, 'high']) if not pd.isna(l_h) else df.loc[i, 'high']
            l_l = min(l_l, df.loc[i, 'low']) if not pd.isna(l_l) else df.loc[i, 'low']
            l_c = df.loc[i, 'close']
        
        if df.loc[i, 'ny_session_start']:
            n_grp = ny_sess.loc[i]
            n_o = df.loc[i, 'open']
        
        asian_high.loc[i] = a_h
        asian_low.loc[i] = a_l
        asian_open.loc[i] = a_o
        asian_close.loc[i] = a_c
        london_high.loc[i] = l_h
        london_low.loc[i] = l_l
        london_open.loc[i] = l_o
        london_close.loc[i] = l_c
        ny_open.loc[i] = n_o
    
    df['asianHigh'] = asian_high
    df['asianLow'] = asian_low
    df['asianOpen'] = asian_open
    df['asianClose'] = asian_close
    df['londonHigh'] = london_high
    df['londonLow'] = london_low
    df['londonOpen'] = london_open
    df['londonClose'] = london_close
    df['nyOpen'] = ny_open
    
    df['asianRange'] = df['asianHigh'] - df['asianLow']
    df['asianMomentum'] = df['asianClose'] - df['asianOpen']
    df['asianBias'] = np.where(df['asianMomentum'] > 0, 1, np.where(df['asianMomentum'] < 0, -1, 0))
    
    df['londonRange'] = df['londonHigh'] - df['londonLow']
    df['londonMomentum'] = df['londonClose'] - df['londonOpen']
    df['londonBias'] = np.where(df['londonMomentum'] > 0, 1, np.where(df['londonMomentum'] < 0, -1, 0))
    
    df['londonBreakAsianHigh'] = df['londonHigh'] > df['asianHigh']
    df['londonBreakAsianLow'] = df['londonLow'] < df['asianLow']
    
    high_2 = df['high'].shift(2)
    low_2 = df['low'].shift(2)
    df['bfvg'] = (df['low'] > high_2) & (~df['low'].isna()) & (~high_2.isna())
    df['sfvg'] = (df['high'] < low_2) & (~df['high'].isna()) & (~low_2.isna())
    
    lastFVG = 0
    bullsharp = False
    bearsharp = False
    confidenceScore = 0
    currentBias = 0
    
    df['bullsharp'] = False
    df['bearsharp'] = False
    df['confidenceScore'] = 0
    df['currentBias'] = 0
    
    prev_asianBias = 0
    prev_londonBias = 0
    prev_londonBreakAsianHigh = False
    prev_londonBreakAsianLow = False
    
    for i in range(1, len(df)):
        idx = df.index[i]
        prev_idx = df.index[i-1]
        
        if df.loc[idx, 'ny_session_start']:
            confidenceScore = 0
            
            asianBias = df.loc[idx, 'asianBias']
            londonBias = df.loc[idx, 'londonBias']
            londonBreakAsianHigh = df.loc[idx, 'londonBreakAsianHigh']
            londonBreakAsianLow = df.loc[idx, 'londonBreakAsianLow']
            londonMomentum = df.loc[idx, 'londonMomentum']
            londonRange = df.loc[idx, 'londonRange']
            
            if asianBias == londonBias and asianBias != 0:
                confidenceScore += int(asianBias * 40)
            
            if londonBreakAsianHigh and londonBias > 0:
                confidenceScore += 25
            
            if londonBreakAsianLow and londonBias < 0:
                confidenceScore -= 25
            
            if abs(londonMomentum) > londonRange * 0.5:
                confidenceScore += int(londonBias * 20)
            
            if londonBias == asianBias and londonBreakAsianHigh and asianBias > 0:
                confidenceScore += 15
            elif londonBias == asianBias and londonBreakAsianLow and asianBias < 0:
                confidenceScore -= 15
            
            if confidenceScore > 30:
                currentBias = 1
            elif confidenceScore < -30:
                currentBias = -1
            else:
                currentBias = 0
        
        prev_bfvg = df.loc[prev_idx, 'bfvg']
        prev_sfvg = df.loc[prev_idx, 'sfvg']
        curr_bfvg = df.loc[idx, 'bfvg']
        curr_sfvg = df.loc[idx, 'sfvg']
        
        bullsharp_i = False
        bearsharp_i = False
        
        if curr_bfvg and lastFVG == -1:
            bullsharp_i = True
            bearsharp_i = False
            lastFVG = 1
        elif curr_sfvg and lastFVG == 1:
            bearsharp_i = True
            bullsharp_i = False
            lastFVG = -1
        elif curr_bfvg:
            lastFVG = 1
        elif curr_sfvg:
            lastFVG = -1
        
        df.loc[idx, 'bullsharp'] = bullsharp_i
        df.loc[idx, 'bearsharp'] = bearsharp_i
        df.loc[idx, 'confidenceScore'] = confidenceScore
        df.loc[idx, 'currentBias'] = currentBias
    
    first_idx = df.index[0]
    df.loc[first_idx, 'confidenceScore'] = 0
    df.loc[first_idx, 'currentBias'] = 0
    df.loc[first_idx, 'bullsharp'] = False
    df.loc[first_idx, 'bearsharp'] = False
    
    in_trading_window = trading_window & ny_session
    df['inTradingWindow'] = in_trading_window
    
    df['bullishEntry'] = df['bullsharp'] & df['inTradingWindow'] & (df['currentBias'] == 1)
    df['bearishEntry'] = df['bearsharp'] & df['inTradingWindow'] & (df['currentBias'] == -1)
    
    entries = []
    trade_num = 1
    position_open = False
    
    for i in range(len(df)):
        idx = df.index[i]
        
        if df.loc[idx, 'bullishEntry'] and not position_open:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df.loc[idx, 'time']),
                'entry_time': datetime.fromtimestamp(df.loc[idx, 'time'], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df.loc[idx, 'close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df.loc[idx, 'close']),
                'raw_price_b': float(df.loc[idx, 'close'])
            })
            trade_num += 1
            position_open = True
        elif df.loc[idx, 'bearishEntry'] and not position_open:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df.loc[idx, 'time']),
                'entry_time': datetime.fromtimestamp(df.loc[idx, 'time'], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df.loc[idx, 'close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df.loc[idx, 'close']),
                'raw_price_b': float(df.loc[idx, 'close'])
            })
            trade_num += 1
            position_open = True
        
        if not df.loc[idx, 'inTradingWindow']:
            position_open = False
    
    return entries