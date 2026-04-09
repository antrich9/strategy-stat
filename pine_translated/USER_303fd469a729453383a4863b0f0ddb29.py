import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Parameters
    wickToBodyRatio = 0.3
    minBodyPct = 0.6
    swingLen = 5
    fvgMaxAge = 50
    useFVGFilter = True
    useSwingFilter = True