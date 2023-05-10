import pandas as pd

s = pd.date_range('2020-01-06', '2020-01-10', freq='10H').to_series()

# create some features based on datetime
features = {
    'dayofweek': s.dt.dayofweek.values,
    'dayofyear': s.dt.dayofyear.values,
    'hour': s.dt.hour.values,
    'is_leap_year': s.dt.is_leap_year.values,
    'quarter': s.dt.quarter.values,
    'weekofyear': s.dt.isocalendar().week.values
}
