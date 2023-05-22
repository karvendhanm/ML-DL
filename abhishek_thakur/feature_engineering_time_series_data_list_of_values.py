# feature engineering
# when dealing with time-series problems, you might have
# features which are not individual values but a list of values.

import numpy as np
from tsfresh.feature_extraction import feature_calculators as fc

x = np.random.randint(low=0, high=100, size=10)

feature_dict = {}

feature_dict['mean'] = np.mean(x)
feature_dict['max'] = np.max(x)
feature_dict['min'] = np.min(x)
feature_dict['std'] = np.std(x)
feature_dict['var'] = np.var(x)
feature_dict['ptp'] = np.ptp(x)         # peak to peak

# percentile features
feature_dict['percentile_10'] = np.percentile(x, 10)
feature_dict['percentile_60'] = np.percentile(x, 60)
feature_dict['percentile_90'] = np.percentile(x, 90)

# quantile features
feature_dict['quantile_10'] = np.quantile(x, 0.10)
feature_dict['quantile_60'] = np.quantile(x, 0.60)
feature_dict['quantile_90'] = np.quantile(x, 0.90)

fc.abs_energy(x)
fc.count_above_mean(x)
fc.count_below_mean(x)
fc.mean_abs_change(x)
fc.mean_change(x)



