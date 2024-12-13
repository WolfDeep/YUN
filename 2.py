import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

data = {
    'x': [0.1, 0.2, 1.0, 1.1, 0.4, 5.0, 5.1, 4.9],
    'y': [0.1, 0.2, 1.1, 1.0, 0.4, 5.1, 5.0, 4.9]
}
df = pd.DataFrame(data)


dbscan = DBSCAN(eps=0.5, min_samples=1, metric='euclidean')
clusters = dbscan.fit_predict(df[['x', 'y']])


df['cluster'] = clusters


print(df)
