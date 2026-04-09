def generate_entries(df):
    # Convert time to datetime in UTC then to London time
    ts = pd.to_datetime(df['time'], unit='s', utc=True)
    london = ts.dt.tz_convert('Europe/London')
    hour = london.dt.hour
    minute = london.dt.minute

    # Define time windows
    morning = ((hour == 7) & (minute >= 45)) | ((hour == 8)) | ((hour == 9) & (minute < 45))
    afternoon = ((hour == 14) & (minute >= 45)) | ((hour == 15)) | ((hour == 16) & (minute < 45))
    isWithinTimeWindow = morning | afternoon