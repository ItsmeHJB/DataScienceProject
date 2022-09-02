import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn import metrics
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

length = 'short'
word_vectors = Word2Vec.load("Models/" + length + ".word2vec.model").wv

model = KMeans(n_clusters=2,
               max_iter=1000,
               random_state=10,
               n_init=50).fit(X=word_vectors.vectors.astype('double'))

# label = model.predict(word_vectors.vectors.astype('double'))
labels = model.labels_

# Has high seperation of clusters due to being close to 1
print(metrics.silhouette_score(word_vectors.vectors.astype('double'), labels))
# High score means better clusters seperation from each other
print(metrics.calinski_harabasz_score(word_vectors.vectors.astype('double'), labels))  #
# Inverse of above, smaller is better
print(metrics.davies_bouldin_score(word_vectors.vectors.astype('double'), labels))

uniq_labels = np.unique(labels)
centroids = model.cluster_centers_

for i in uniq_labels:
    plt.scatter(word_vectors.vectors.astype('double')[labels == i, 0],
                word_vectors.vectors.astype('double')[labels == i, 1], label=i)
plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
plt.legend()
plt.show()
