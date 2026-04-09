import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Convert time to datetime
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df['date'] = df['datetime'].dt.date
    
    # Get previous day high/low
    # Group by date and get high/low
    daily_stats = df.groupby('date')['high'].max().reset_index()
    daily_stats.columns = ['date', 'daily_high']
    daily_lows = df.groupby('date')['low'].min().reset_index()
    daily_lows.columns = ['date', 'daily_low']
    daily_stats = daily_stats.merge(daily_lows, on='date')
    
    # Previous day high/low: shift to get previous day values
    daily_stats['prev_day_high'] = daily_stats['daily_high'].shift(1)
    daily_stats['prev_day_low'] = daily_stats['daily_low'].shift(1)
    
    # Merge back to main dataframe
    df = df.merge(daily_stats[['date', 'prev_day_high', 'prev_day_low']], on='date', how='left')
    
    # Current day high/low: use expanding window within each day
    df['current_day_high'] = df.groupby('date')['high'].transform(lambda x: x.cummin())
    df['current_day_low'] = df.groupby('date')['low'].transform(lambda x: x.cummax())
    
    # Wait, that's not right. Current day high should be the high so far in the current day
    # Actually, for the strategy, we need the high/low of the current day up to the current bar
    # But looking at the Pine code: currentDayHigh = request.security(..., '240', high, ...)
    # This suggests using a higher timeframe (240min = 4H) for the current day
    # Since we don't have 4H data in the dataframe, we'll use the daily data but need to be careful
    
    # Actually, for current day high/low in Pine, it's the high/low of the current day so far
    # In Python, we can calculate expanding max/min within each day
    
    # Let me recalculate
    df['current_day_high'] = df.groupby('date')['high'].cummax()
    df['current_day_low'] = df.groupby('date')['low'].cummin()
    
    # Time windows (London time)
    # Morning: 07:45 to 09:45 London
    # Afternoon: 15:45 to 16:45 London
    # Since data is in UTC (based on unix timestamps), and London time can be BST or GMT
    # We need to handle timezone conversion
    # For simplicity, assume the data is in UTC and London time is UTC+0 or UTC+1 depending on DST
    
    # Simplified: Check if in time window (ignoring DST for now, or handling it)
    # Actually, let's just check the hour and minute in UTC, but adjust for London
    # London time = UTC in winter, UTC+1 in summer
    # We can check if it's summer time (julian day > 80 and < 280 approx)
    
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear
    
    # Simple DST check: if month > 3 and month < 10, likely DST (but this is approximate)
    # or use proper DST detection
    # Let's use a simple approach: UTC+1 if DST
    
    # Actually, let's convert to London time explicitly
    # But we don't have the timezone library in requirements... wait, we can use pytz or datetime with timezone
    # The function signature doesn't forbid pytz, but it says "Use ONLY pandas and numpy"
    # So we can't use pytz. We need to handle it manually.
    
    # Manual timezone handling:
    # London is UTC+0 in winter, UTC+1 in summer
    # DST starts: last Sunday of March (clocks go forward)
    # DST ends: last Sunday of October (clocks go back)
    
    # This is complex. Let's simplify: assume London is always UTC+0 for now
    # Or, let's check the hour in UTC and compare with London time window
    # Morning window 07:45-09:45 London = 07:45-09:45 UTC if GMT, or 06:45-08:45 UTC if BST
    # Let's assume data is in UTC and adjust accordingly
    
    # Actually, looking at the Pine code: it uses timestamp("Europe/London", ...)
    # This means the strategy was written assuming the chart is in the exchange timezone or similar
    # For the Python conversion, we need to know what timezone the data is in
    # The user says: "time(int unix ts)" which is typically UTC
    
    # Let's assume the data is in UTC and the time window checks should be in UTC
    # But the Pine code explicitly uses Europe/London timestamps
    # This means we need to convert to London time
    
    # Let's add a London timezone offset column
    # DST: March last Sunday to October last Sunday
    def get_london_offset(row):
        month = row['month']
        day = row['datetime'].day
        dayofweek = row['datetime'].dayofweek  # Monday=0, Sunday=6
        
        if month < 3 or month > 10:
            return 0  # GMT
        elif month > 3 and month < 10:
            return 1  # BST
        else:  # March or October
            if month == 3:
                # Last Sunday is day 25 - (6 - dayofweek) if dayofweek > 0, or 31 if dayofweek == 0 but after 25
                # Actually, last Sunday = 31 - (weekday of last day or something)
                # Simplified: if month == 3 and day >= 25 and dayofweek == 6: DST starts
                # But we need to check if it's the last Sunday
                # Last Sunday of March: 25-31
                if dayofweek == 6 and day >= 25:
                    return 1  # BST starts this day
                else:
                    return 0  # GMT
            else:  # October
                # Last Sunday of October: 25-31
                if dayofweek == 6 and day >= 25:
                    return 0  # GMT starts this day
                else:
                    return 1  # BST
                    
    # This is getting complex. Let's simplify further.
    # For the purpose of this conversion, let's assume the data is in the exchange timezone or UTC
    # and the time checks are relative to that timezone.
    # Actually, let's just check the hour/minute in UTC and see if it matches the London time windows
    # adjusted for the fact that the timestamp is in UTC.
    
    # Wait, let's look at the Pine code again:
    # london_start_morning = timestamp("Europe/London", year, month, dayofmonth, 7, 45)
    # This creates a timestamp in London time.
    # Then it compares with `time` which is the bar time in the chart's timezone (usually exchange timezone)
    # So the data in Python (unix timestamp) is likely in the exchange timezone or UTC
    # We need to know the timezone of the data
    
    # Given the complexity, let's make a reasonable assumption:
    # The data is in UTC (standard for unix timestamps)
    # We need to check if the UTC time corresponds to the London time windows
    # Morning window: 07:45-09:45 London = 06:45-08:45 UTC (Mar-Oct), 07:45-09:45 UTC (Nov-Feb)
    # Afternoon window: 15:45-16:45 London = 14:45-15:45 UTC (Mar-Oct), 15:45-16:45 UTC (Nov-Feb)
    
    # Let's implement a function to check if in time window, accounting for DST
    # DST offset: 1 hour from last Sunday March to last Sunday October
    
    # Calculate DST offset
    def is_dst(datetime_series):
        # Last Sunday of March and October
        # This is tricky to calculate precisely without calendar knowledge
        # Let's use a simpler approximation or manual calculation
        
        # Actually, let's just check if month is between 4 and 9 (inclusive) for BST
        # Or between 10 and 3 for GMT, with edge cases
        month = datetime_series.month
        day = datetime_series.day
        dayofweek = datetime_series.dayofweek
        
        # If March and day >= 25 and Sunday
        # If October and day < 25 or not Sunday
        
        # Simplified: BST if month > 3 and month < 10
        # OR month == 3 and day >= 25 and dayofweek == 6
        # OR month == 10 and day < 25 or (day >= 25 and dayofweek != 6)
        
        # This is still complex. Let's use a lookup table or simpler logic.
        # Let's assume for simplicity that we don't have accurate DST handling,
        # or let's implement it properly.
        
        pass
    
    # Okay, let's just implement a reasonable approximation:
    # BST (UTC+1) if month is April to September, or March after last Sunday, or October before last Sunday
    
    # Actually, for the code to be correct, we need to handle DST properly.
    # Let's calculate the last Sunday of March and October for each year.
    
    # For simplicity, let's assume the data spans a limited time period or use a simple heuristic.
    # Or, let's just check UTC time and subtract 1 if it could be BST.
    
    # Let's implement proper DST:
    # Find last Sunday of March/October for the year
    def last_sunday(year, month):
        if month == 3 or month == 10:
            # Find the last day of the month
            if month == 3:
                last_day = 31  # March has 31 days
            else:
                last_day = 31  # October has 31 days
            
            # Find what day of week the last day is
            # January 1, year is dayofweek
            # December 31, year is dayofweek + (365 or 366 if leap year) % 7
            # This is getting too complex without datetime utilities
            
            # Let's use a simpler approach: assume BST if month in [4,5,6,7,8,9] or (month==3 and day>=25) or (month==10 and day<25)
            pass
    
    # Okay, let's use pandas: pd.Timestamp has tzinfo
    # We can convert the datetime to London timezone if we had pytz, but we don't.
    # So let's manually calculate the London time.
    
    # London timezone offset:
    # Returns 1 if BST (summer), 0 if GMT (winter)
    def get_dst_offset(dt):
        # Algorithm:
        # If month < 3 or month > 10: offset = 0 (GMT)
        # If month > 3 and month < 10: offset = 1 (BST)
        # If month == 3:
        #   If day > 25: offset = 1
        #   If day == 25: check if it's before or after 1am GMT (the switch happens at 1am)
        #   If day < 25: offset = 0
        # If month == 10:
        #   If day < 25: offset = 1
        #   If day == 25: check if it's before or after 1am BST (the switch happens at 1am)
        #   If day > 25: offset = 0
        
        month = dt.month
        day = dt.day
        
        if month < 3 or month > 10:
            return 0
        elif month > 3 and month < 10:
            return 1
        elif month == 3:
            # DST starts: last Sunday of March at 1am GMT
            # Find last Sunday
            # Days in March: 31
            # If dayofweek of March 31 is X, then last Sunday is 31 - X
            import datetime
            date_obj = datetime.date(dt.year, 3, 31)
            dayofweek = date_obj.weekday()  # Monday=0
            days_since_last_sunday = (dayofweek + 1) % 7  # Convert to Sunday=6 format
            last_sunday = 31 - days_since_last_sunday
            
            if day < last_sunday:
                return 0
            elif day > last_sunday:
                return 1
            else:  # day == last_sunday
                # Before 1am GMT? Let's assume it's after for simplicity, or check hour
                # If hour < 1: return 0, else: return 1
                # But we don't have hour here, so let's assume after
                return 1
        else:  # month == 10
            # DST ends: last Sunday of October at 1am GMT (so becomes UTC+0)
            date_obj = datetime.date(dt.year, 10, 31)
            dayofweek = date_obj.weekday()
            days_since_last_sunday = (dayofweek + 1) % 7
            last_sunday = 31 - days_since_last_sunday
            
            if day < last_sunday:
                return 1
            elif day > last_sunday:
                return 0
            else:  # day == last_sunday
                return 0  # Assume after the switch
    
    # Actually, let's simplify: we can check if (month, day) is in the BST period
    # BST starts: last Sunday March
    # BST ends: last Sunday October
    
    # Let's calculate this properly
    import datetime
    
    def get_london_offset_ts(ts):
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        month = dt.month
        day = dt.day
        
        if month < 3 or month > 10:
            return 0
        elif month > 3 and month < 10:
            return 1
        else:
            # March or October
            if month == 3:
                # Find last Sunday
                last_day = 31
                # weekday() returns 0=Monday, 6=Sunday for the last day
                wd = datetime.date(dt.year, 3, last_day).weekday()
                days_to_subtract = (wd + 1) % 7  # Days since last Sunday
                last_sunday = last_day - days_to_subtract
                
                if day < last_sunday:
                    return 0
                elif day > last_sunday:
                    return 1
                else:  # day == last_sunday
                    return 1  # After 1am UTC (when DST starts)
            else:  # October
                last_day = 31
                wd = datetime.date(dt.year, 10, last_day).weekday()
                days_to_subtract = (wd + 1) % 7
                last_sunday = last_day - days_to_subtract
                
                if day < last_sunday:
                    return 1
                elif day > last_sunday:
                    return 0
                else:  # day == last_sunday
                    return 0  # After 1am UTC (when DST ends, so back to GMT)
    
    # Now let's apply this to get London time
    # London time = UTC + offset
    df['london_offset'] = df['time'].apply(get_london_offset_ts)
    
    # Now check time windows in London time
    # We need to convert UTC time to London time
    # London time hour = UTC hour + offset
    df['london_hour'] = (df['hour'] + df['london_offset']) % 24
    df['london_minute'] = df['minute']  # Minutes are the same
    
    # Morning window: 07:45 to 09:45
    # Afternoon window: 15:45 to 16:45
    
    def in_time_window(row):
        hour = row['london_hour']
        minute = row['london_minute']
        
        # Morning: 07:45 to 09:45
        if (hour == 7 and minute >= 45) or (8 <= hour <= 9) or (hour == 9 and minute <= 45):
            if hour == 9 and minute > 45:
                return False
            return True
        
        # Afternoon: 15:45 to 16:45
        if (hour == 15 and minute >= 45) or (hour == 16 and minute <= 45):
            if hour == 16 and minute > 45:
                return False
            return True
        
        return False
    
    df['in_time_window'] = df.apply(in_time_window, axis=1)
    
    # Now the conditions for entries
    # previousDayHighTaken = high > prevDayHigh
    # previousDayLowTaken = low < prevDayLow
    df['previousDayHighTaken'] = df['high'] > df['prev_day_high']
    df['previousDayLowTaken'] = df['low'] < df['prev_day_low']
    
    # flagpdh and flagpdl
    # flagpdh = previousDayHighTaken and currentDayLow > prevDayLow
    # flagpdl = previousDayLowTaken and currentDayHigh < prevDayHigh
    
    df['flagpdh'] = df['previousDayHighTaken'] & (df['current_day_low'] > df['prev_day_low'])
    df['flagpdl'] = df['previousDayLowTaken'] & (df['current_day_high'] < df['prev_day_high'])
    
    # Entries:
    # Long: flagpdl and in_time_window
    # Short: flagpdh and in_time_window
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Check for NaN in required columns
        if pd.isna(row['prev_day_high']) or pd.isna(row['prev_day_low']):
            continue
        
        # Check entry conditions
        if row['in_time_window']:
            if row['flagpdl']:
                # Long entry
                entry = {
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(row['time']),
                    'entry_time': datetime.fromtimestamp(row['time'] / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': row['close'],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': row['close'],
                    'raw_price_b': row['close']
                }
                entries.append(entry)
                trade_num += 1
            
            if row['flagpdh']:
                # Short entry
                entry = {
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(row['time']),
                    'entry_time': datetime.fromtimestamp(row['time'] / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': row['close'],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': row['close'],
                    'raw_price_b': row['close']
                }
                entries.append(entry)
                trade_num += 1
    
    return entries