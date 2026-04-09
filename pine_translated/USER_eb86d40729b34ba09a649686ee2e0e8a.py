import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    Entry logic only - no exits, stops, or position management.
    """
    # ========== PREPROCESSING ==========
    df = df.copy()
    df = df.reset_index(drop=True)
    
    # Convert time to datetime for reference
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # ========== INDICATOR PARAMETERS ==========
    # HTF Structure parameters (mirrors Pine Script)
    PP_htf = 6  # Pivot lookback/forward
    
    # ========== WILDER RSI ==========
    def wilder_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # ========== WILDER ATR ==========
    def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    # ========== PIVOT DETECTION ==========
    def pivothigh(source: pd.Series, left_bars: int, right_bars: int) -> pd.Series:
        ph = pd.Series(np.nan, index=source.index)
        for i in range(left_bars, len(source) - right_bars):
            window = source.iloc[i - left_bars : i + right_bars + 1]
            if source.iloc[i] == window.max():
                ph.iloc[i] = source.iloc[i]
        return ph
    
    def pivotlow(source: pd.Series, left_bars: int, right_bars: int) -> pd.Series:
        pl = pd.Series(np.nan, index=source.index)
        for i in range(left_bars, len(source) - right_bars):
            window = source.iloc[i - left_bars : i + right_bars + 1]
            if source.iloc[i] == window.min():
                pl.iloc[i] = source.iloc[i]
        return pl
    
    # Calculate pivots on HTF (using close as proxy - actual would need multi-timeframe)
    high_pivot = pivothigh(df['high'], PP_htf, PP_htf)
    low_pivot = pivotlow(df['low'], PP_htf, PP_htf)
    
    # ========== ZIGZAG ARRAYS (SIMPLIFIED) ==========
    # Simplified ZigZag tracking
    zz_types = []  # 'H', 'L', 'HH', 'HL', 'LH', 'LL'
    zz_values = []
    zz_indices = []
    
    # Structure tracking
    Major_HighLevel = np.nan
    Major_LowLevel = np.nan
    Minor_HighLevel = np.nan
    Minor_LowLevel = np.nan
    
    # Signal booleans
    Bullish_Major_ChoCh = False
    Bearish_Major_ChoCh = False
    Bullish_Major_BoS = False
    Bearish_Major_BoS = False
    Bullish_Minor_ChoCh = False
    Bearish_Minor_ChoCh = False
    Bullish_Minor_BoS = False
    Bearish_Minor_BoS = False
    
    # Track last structure direction
    last_structure = 'none'  # 'bullish', 'bearish'
    
    # ========== STRUCTURE DETECTION LOGIC ==========
    for i in range(PP_htf * 2, len(df)):
        # Update ZigZag
        hp = high_pivot.iloc[i]
        lp = low_pivot.iloc[i]
        
        if not pd.isna(hp) and not pd.isna(lp):
            if len(zz_values) == 0:
                zz_types.append('H' if hp >= lp else 'L')
                zz_values.append(hp if hp >= lp else lp)
                zz_indices.append(i)
            else:
                last_type = zz_types[-1]
                last_val = zz_values[-1]
                
                if last_type in ['L', 'LL']:
                    if lp < last_val:
                        zz_types[-1] = 'LL' if len(zz_values) > 2 and zz_values[-2] < lp else 'L'
                        zz_values[-1] = lp
                        zz_indices[-1] = i
                        Major_LowLevel = lp
                    else:
                        zz_types.append('HH' if len(zz_values) > 1 and zz_values[-2] < hp else 'LH')
                        zz_values.append(hp)
                        zz_indices.append(i)
                        Major_HighLevel = hp
                elif last_type in ['H', 'HH']:
                    if hp > last_val:
                        zz_types[-1] = 'HH' if len(zz_values) > 2 and zz_values[-2] < hp else 'H'
                        zz_values[-1] = hp
                        zz_indices[-1] = i
                        Major_HighLevel = hp
                    else:
                        zz_types.append('HL' if len(zz_values) > 1 and zz_values[-2] < lp else 'LL')
                        zz_values.append(lp)
                        zz_indices.append(i)
                        Major_LowLevel = lp
                elif last_type == 'LH':
                    if lp < last_val:
                        zz_types.append('LL' if len(zz_values) > 2 and zz_values[-2] < lp else 'L')
                        zz_values.append(lp)
                        zz_indices.append(i)
                        Major_LowLevel = lp
                    elif hp > last_val:
                        if df['close'].iloc[i] < last_val:
                            zz_types[-1] = 'HH'
                            zz_values[-1] = hp
                            zz_indices[-1] = i
                            Major_HighLevel = hp
                        else:
                            zz_types.append('LL' if len(zz_values) > 2 and zz_values[-2] < lp else 'L')
                            zz_values.append(lp)
                            zz_indices.append(i)
                            Major_LowLevel = lp
                elif last_type == 'HL':
                    if lp > last_val:
                        zz_types.append('HH' if len(zz_values) > 2 and zz_values[-2] < hp else 'H')
                        zz_values.append(hp)
                        zz_indices.append(i)
                        Major_HighLevel = hp
                    elif lp < last_val:
                        if df['close'].iloc[i] > last_val:
                            zz_types[-1] = 'LL'
                            zz_values[-1] = lp
                            zz_indices[-1] = i
                            Major_LowLevel = lp
                        else:
                            zz_types.append('HH' if len(zz_values) > 2 and zz_values[-2] < hp else 'H')
                            zz_values.append(hp)
                            zz_indices.append(i)
                            Major_HighLevel = hp
        
        # Structure detection based on ZigZag patterns
        if len(zz_types) >= 3:
            current_type = zz_types[-1]
            prev_type = zz_types[-2]
            prev_val = zz_values[-2]
            curr_val = zz_values[-1]
            
            # Major Structure Detection
            if len(zz_values) >= 4:
                # Bullish BoS: Higher highs and higher lows
                if (zz_values[-3] < zz_values[-1] and 
                    zz_values[-4] < zz_values[-2] and
                    not Bullish_Major_BoS):
                    Bullish_Major_BoS = True
                    Bearish_Major_BoS = False
                    last_structure = 'bullish'
                
                # Bearish BoS: Lower highs and lower lows
                if (zz_values[-3] > zz_values[-1] and 
                    zz_values[-4] > zz_values[-2] and
                    not Bearish_Major_BoS):
                    Bearish_Major_BoS = True
                    Bullish_Major_BoS = False
                    last_structure = 'bearish'
            
            # CHoCH detection (change within structure)
            if len(zz_values) >= 3:
                # Bullish CHoCH
                if current_type in ['LH', 'H'] and prev_type in ['L', 'LL']:
                    if prev_val < curr_val and not Bullish_Major_ChoCh:
                        Bullish_Major_ChoCh = True
                        Bearish_Major_ChoCh = False
                
                # Bearish CHoCH
                if current_type in ['HL', 'L'] and prev_type in ['H', 'HH']:
                    if prev_val > curr_val and not Bearish_Major_ChoCh:
                        Bearish_Major_ChoCh = True
                        Bullish_Major_ChoCh = False
    
    # ========== FVG DETECTION ==========
    def detect_fvg(df: pd.DataFrame) -> tuple:
        """
        Detect Fair Value Gaps (3-candle imbalance)
        Returns (bullish_fvg, bearish_fvg, fvg_high, fvg_low)
        """
        bullish_fvg = pd.Series(False, index=df.index)
        bearish_fvg = pd.Series(False, index=df.index)
        fvg_high = pd.Series(np.nan, index=df.index)
        fvg_low = pd.Series(np.nan, index=df.index)
        
        for i in range(2, len(df)):
            # Bullish FVG: gap between candle 1 low and candle 3 high
            candle1_low = df['low'].iloc[i-2]
            candle3_high = df['high'].iloc[i]
            candle2_high = df['high'].iloc[i-1]
            candle2_low = df['low'].iloc[i-1]
            
            if candle1_low > candle3_high:
                bullish_fvg.iloc[i] = True
                fvg_low.iloc[i] = candle3_high
                fvg_high.iloc[i] = candle1_low
            
            # Bearish FVG: gap between candle 1 high and candle 3 low
            if candle1_low > candle3_high and candle2_high > candle3_high:
                # Check for bearish FVG condition
                pass
            
            # More robust FVG detection
            # Bullish FVG: candle 1 high < candle 3 low (with overlap check)
            if df['high'].iloc[i-2] < df['low'].iloc[i]:
                bullish_fvg.iloc[i] = True
                fvg_low.iloc[i] = df['low'].iloc[i]
                fvg_high.iloc[i] = df['high'].iloc[i-2]
            
            # Bearish FVG: candle 1 low > candle 3 high
            if df['low'].iloc[i-2] > df['high'].iloc[i]:
                bearish_fvg.iloc[i] = True
        
        return bullish_fvg, bearish_fvg, fvg_high, fvg_low
    
    bullish_fvg, bearish_fvg, fvg_high, fvg_low = detect_fvg(df)
    
    # ========== BUILD ENTRY SIGNALS ==========
    entries = []
    trade_num = 1
    
    # Track in-trade state
    in_long_trade = False
    in_short_trade = False
    
    # Structure state variables
    bullish_structure = False
    bearish_structure = False
    
    # Recalculate structure state based on recent pivots for entry decisions
    for i in range(PP_htf * 2, len(df)):
        # Calculate current structure state
        current_bullish = False
        current_bearish = False
        
        # Look back for recent structure
        lookback = min(20, i)
        recent_highs = df['high'].iloc[i-lookback:i].values
        recent_lows = df['low'].iloc[i-lookback:i].values
        
        if len(recent_highs) >= 5:
            hh_count = sum(1 for j in range(2, len(recent_highs)-1) if recent_highs[j] > recent_highs[j-1] and recent_highs[j] > recent_highs[j+1])
            hl_count = sum(1 for j in range(2, len(recent_lows)-1) if recent_lows[j] > recent_lows[j-1] and recent_lows[j] > recent_lows[j+1])
            lh_count = sum(1 for j in range(2, len(recent_highs)-1) if recent_highs[j] < recent_highs[j-1] and recent_highs[j] < recent_highs[j+1])
            ll_count = sum(1 for j in range(2, len(recent_lows)-1) if recent_lows[j] < recent_lows[j-1] and recent_lows[j] < recent_lows[j+1])
            
            # Bullish structure: HHs and HLs
            if hh_count >= 2 and hl_count >= 2:
                current_bullish = True
            
            # Bearish structure: LHs and LLs
            if lh_count >= 2 and ll_count >= 2:
                current_bearish = True
        
        # ========== ENTRY CONDITIONS ==========
        # Long Entry: Bullish FVG in bullish structure
        long_condition = (
            bullish_fvg.iloc[i] and 
            current_bullish and 
            not in_long_trade and 
            not in_short_trade
        )
        
        # Short Entry: Bearish FVG in bearish structure
        short_condition = (
            bearish_fvg.iloc[i] and 
            current_bearish and 
            not in_short_trade and 
            not in_long_trade
        )
        
        if long_condition:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            
            trade_num += 1
            in_long_trade = True
        
        elif short_condition:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            
            trade_num += 1
            in_short_trade = True
        
        # Reset trade state when opposite signal appears (simple exit logic)
        if in_long_trade and bearish_fvg.iloc[i] and current_bearish:
            in_long_trade = False
        
        if in_short_trade and bullish_fvg.iloc[i] and current_bullish:
            in_short_trade = False
    
    return entries