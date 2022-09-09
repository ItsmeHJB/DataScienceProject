import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt


length = "high"
version = "1"
word_vectors = Word2Vec.load("Models/"+length+".word2vec.model").wv

model = KMeans(n_clusters=2,
               max_iter=1000,
               random_state=10,
               n_init=50).fit(X=word_vectors.vectors.astype('double'))

labels = model.labels_

# Has high seperation of clusters due to being close to 1
print(metrics.silhouette_score(word_vectors.vectors.astype('double'), labels))
# High score means better clusters seperation from each other
print(metrics.calinski_harabasz_score(word_vectors.vectors.astype('double'), labels))  #
# Inverse of above, smaller is better
# print(metrics.davies_bouldin_score(word_vectors.vectors.astype('double'), labels))

uniq_labels = np.unique(labels)
centroids = model.cluster_centers_

# for i in uniq_labels:
#     plt.scatter(word_vectors.vectors.astype('double')[labels == i, 0],
#                 word_vectors.vectors.astype('double')[labels == i, 1], label=i)
# plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
# plt.legend()
# plt.show()

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

word_vectors.similar_by_vector(model.cluster_centers_[1], topn=10, restrict_vocab=None)

positive_cluster_index = 1
positive_cluster_center = model.cluster_centers_[positive_cluster_index]
negative_cluster_center = model.cluster_centers_[1 - positive_cluster_index]

words = pd.DataFrame(word_vectors.index_to_key)
words.columns = ['words']
words['vectors'] = words.words.apply(lambda x: word_vectors[f'{x}'])
words['cluster'] = words.vectors.apply(lambda x: model.predict([np.array(x)])) # This can take a while
words.cluster = words.cluster.apply(lambda x: x[0])

words['cluster_value'] = [1 if i == positive_cluster_index else -1 for i in words.cluster]
words['closeness_score'] = words.apply(lambda x: 1 / (model.transform([x.vectors]).min()), axis=1)
words['sentiment_coeff'] = words.closeness_score * words.cluster_value

print(words.head(10))

final_file = pd.read_csv('cleaned_dataset.csv')

sentiment_map = words.reset_index(drop=True).copy()
sentiment_dict = dict(zip(sentiment_map.words.values, sentiment_map.sentiment_coeff.values))

file_weighting = final_file.copy()

tfidf = TfidfVectorizer(tokenizer=lambda y: y.split(), norm=None)
tfidf.fit(file_weighting.sent)
features = pd.Series(tfidf.get_feature_names())
transformed = tfidf.transform(file_weighting.sent)


def create_tfidf_dictionary(x, transformed_file, features):
    """
    create dictionary for each input sentence x, where each word has assigned its tfidf score

    inspired  by function from this wonderful article:
    https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34

    x - row of dataframe, containing sentences, and their indexes,
    transformed_file - all sentences transformed with TfidfVectorizer
    features - names of all words in corpus used in TfidfVectorizer

    """
    vector_coo = transformed_file[x.name].tocoo()
    vector_coo.col = features.iloc[vector_coo.col].values
    dict_from_coo = dict(zip(vector_coo.col, vector_coo.data))
    return dict_from_coo


def replace_tfidf_words(x, transformed_file, features):
    """
    replacing each word with it's calculated tfidf dictionary with scores of each word
    x - row of dataframe, containing sentences, and their indexes,
    transformed_file - all sentences transformed with TfidfVectorizer
    features - names of all words in corpus used in TfidfVectorizer
    """
    dictionary = create_tfidf_dictionary(x, transformed_file, features)
    return list(map(lambda y: dictionary[f'{y}'], x.sent.split()))


replaced_tfidf_scores = file_weighting.apply(lambda x: replace_tfidf_words(x, transformed, features),
                                             axis=1)  # this step takes a while


def replace_sentiment_words(word, sentiment_dict):
    """
    replacing each word with its associated sentiment score from sentiment dict
    """
    try:
        out = sentiment_dict[word]
    except KeyError:
        out = 0
    return out


replaced_closeness_scores = file_weighting.sent.apply(
    lambda x: list(map(lambda y: replace_sentiment_words(y, sentiment_dict), x.split())))

replacement_df = pd.DataFrame(data=[replaced_closeness_scores, replaced_tfidf_scores, file_weighting.sent]).T
replacement_df.columns = ['sentiment_coeff', 'tfidf_scores', 'sentence']
replacement_df['sentiment_rate'] = replacement_df.apply(
    lambda x: np.array(x.loc['sentiment_coeff']) @ np.array(x.loc['tfidf_scores']), axis=1)
replacement_df['prediction'] = (replacement_df.sentiment_rate > 0).astype('int8')

print(replacement_df.value_counts(['prediction']))

replacement_df.to_csv("Output/KMeans"+length+version+"_predictions.csv", index=False)
