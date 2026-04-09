import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.
    """
    # Input parameters
    liqLen = 7
    liqMar = 10 / 6.9
    atrLength = 14
    fvgWaitBars = 10
    fvgMinTicks = 3
    waitForFVG = True
    useSession = False
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate ATR using Wilder's method
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atrLength, adjust=False).mean()
    
    # Calculate tolerance
    tolerance = atr / liqMar
    
    # Detect pivot highs
    pivot_high = (high.rolling(window=liqLen*2+1, center=True).max() == high) & (high.shift(liqLen) == high)
    pivot_high_vals = high * pivot_high
    
    # Detect pivot lows
    pivot_low = (low.rolling(window=liqLen*2+1, center=True).min() == low) & (low.shift(liqLen) == low)
    pivot_low_vals = low * pivot_low
    
    # Store zone information
    zones = []
    
    # Track last breach info
    last_bsl_breach_bar = -999
    last_bsl_breach_price = np.nan
    last_ssl_breach_bar = -999
    last_ssl_breach_price = np.nan
    
    entries = []
    
    # Iterate through bars
    for i in range(len(df)):
        current_high = high.iloc[i]
        current_low = low.iloc[i]
        current_atr = atr.iloc[i]
        current_tolerance = current_atr / liqMar
        
        # Check for new pivots and update zones
        if i >= liqLen:
            # Check for pivot high
            if pivot_high_vals.iloc[i] > 0:
                # Create new buyside zone
                zone_price = pivot_high_vals.iloc[i]
                zones.append({
                    'type': 'bsl',
                    'price': zone_price,
                    'bar': i - liqLen,
                    'active': True
                })
            
            # Check for pivot low
            if pivot_low_vals.iloc[i] > 0:
                # Create new sellside zone
                zone_price = pivot_low_vals.iloc[i]
                zones.append({
                    'type': 'ssl',
                    'price': zone_price,
                    'bar': i - liqLen,
                    'active': True
                })
        
        # Check for breaches
        for zone in zones:
            if zone['active']:
                if zone['type'] == 'bsl' and current_high > zone['price'] + current_tolerance:
                    zone['active'] = False
                    last_bsl_breach_bar = i
                    last_bsl_breach_price = zone['price']
                elif zone['type'] == 'ssl' and current_low < zone['price'] - current_tolerance:
                    zone['active'] = False
                    last_ssl_breach_bar = i
                    last_ssl_breach_price = zone['price']
        
        # Check for entry conditions
        if useSession and not inSession(df['time'].iloc[i]):
            continue
        
        bsl_breach_valid = i - last_bsl_breach_bar <= fvgWaitBars if last_bsl_breach_bar > -999 else False
        ssl_breach_valid = i - last_ssl_breach_bar <= fvgWaitBars if last_ssl_breach_bar > -999 else False
        
        if ssl_breach_valid and waitForFVG:
            bullish_fvg = low.iloc[i] > high.iloc[i-2]
            if bullish_fvg:
                entries.append({
                    'type': 'long',
                    'entry_bar': i,
                    'entry_price': close.iloc[i],
                    'stop_loss': low.iloc[i] - current_tolerance,
                    'take_profit': close.iloc[i] + 2 * current_tolerance
                })
        
        if bsl_breach_valid and waitForFVG:
            bearish_fvg = high.iloc[i] < low.iloc[i-2]
            if bearish_fvg:
                entries.append({
                    'type': 'short',
                    'entry_bar': i,
                    'entry_price': close.iloc[i],
                    'stop_loss': high.iloc[i] + current_tolerance,
                    'take_profit': close.iloc[i] - 2 * current_tolerance
                })
    
    return entries

def inSession(timestamp):
    from datetime import datetime
    dt = datetime.fromtimestamp(timestamp)
    hour = dt.hour
    return (hour >= 9 and hour < 16) or (hour >= 20 and hour < 23)