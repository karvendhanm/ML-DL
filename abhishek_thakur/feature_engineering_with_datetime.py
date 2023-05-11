import pandas as pd

s = pd.date_range('2020-01-06', '2020-02-07', freq='7D').to_series()

# create some features based on datetime
features = {
    'dayofweek': s.dt.dayofweek.values,
    'dayofyear': s.dt.dayofyear.values,
    'hour': s.dt.hour.values,
    'is_leap_year': s.dt.is_leap_year.values,
    'quarter': s.dt.quarter.values,
    'weekofyear': s.dt.isocalendar().week.values
}

_dict = {
    'date': list(s),  # a series of time stamps
    'customer_id': [146361, 180838, 157857, 159772, 80014],
    'cat1': [2, 4, 3, 5, 3],
    'cat2': [2, 1, 3, 1, 2],
    'cat3': [0, 0, 1, 1, 1],
    'num1': [-0.518679, 0.415853, -2.061687, -0.276558, -1.456827]
}

df = pd.DataFrame.from_dict(_dict)
print(df.dtypes)