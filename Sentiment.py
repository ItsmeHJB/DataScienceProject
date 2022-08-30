# Inspiration: https://github.com/rafaljanwojcik/Unsupervised-Sentiment-Analysis/blob/master/preprocessing_and_embeddings/Preprocessing_and_Embeddings.ipynb

import pandas as pd
from re import sub
import slur_detection

file = pd.read_csv("matches.csv", sep=',', header=None, names=['word', 'url'])
print(file)
dataset = file.drop_duplicates().reset_index(drop=True)

# get text from url -> decode and make useable
# find sentence where word is used. either between \n or .'s
# throw to func
# replace non-alphanumerical signs, punctatuion and duplicated white space with single white space


def text_to_word_list(text):
    text = sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", text)
    text = sub(r"\+", " plus ", text)
    text = sub(r",", " ", text)
    text = sub(r"\.", " ", text)
    text = sub(r"!", " ! ", text)
    text = sub(r"\?", " ? ", text)
    text = sub(r"'", " ", text)
    text = sub(r":", " : ", text)
    text = sub(r"\s{2,}", " ", text)

    text = text.split()

    return text


print(dataset)

for index, row in dataset.iterrows():
    wet_file = slur_detection.get_file(row['url'])
    wet_record = slur_detection.get_record_with_header(
        wet_file,
        header='WARC-Identified-Content-Language',
        value="eng"
    )
    full_text = slur_detection.get_WET_text(wet_record)

    print([sentence + '.' for sentence in full_text.split('.') if row['word'] in sentence])
