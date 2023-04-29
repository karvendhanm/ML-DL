# t-distributed stochastic neighbour embedding (t-SNE)

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import datasets
from sklearn import manifold

data = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
pixel_values, targets = data
targets = targets.astype(int)

# visualizing a single image
single_image = np.array(pixel_values.iloc[1, :]).reshape(28, 28)
plt.imshow(single_image, cmap='gray')

tsne = manifold.TSNE(n_components=2, random_state=42)
transfromed_data = tsne.fit_transform(pixel_values.iloc[:3000, :])

tsne_df = pd.DataFrame(np.column_stack((transfromed_data, targets[:3000])),
                       columns=['x', 'y', 'targets'])
tsne_df.loc[:, 'targets'] = tsne_df.targets.astype(int)

grid = sns.FacetGrid(tsne_df, hue='targets')
grid.map(plt.scatter, 'x', 'y').add_legend()








