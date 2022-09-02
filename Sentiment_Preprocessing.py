# Inspiration: https://github.com/rafaljanwojcik/Unsupervised-Sentiment-Analysis/blob/master/preprocessing_and_embeddings/Preprocessing_and_Embeddings.ipynb
import re
import pandas as pd
from re import sub
import requests
from warcio.archiveiterator import ArchiveIterator
from gensim.models.phrases import Phrases, Phraser
from time import time
from gensim.models import Word2Vec
import multiprocessing


def get_file(url):
    return requests.get(url, stream=True)


def wrapper(gen):
    while True:
        try:
            yield next(gen)
        except StopIteration:
            break
        except Exception as e:
            print(e)


def get_record_with_header(warc_file, header, value):
    iterobject = wrapper(ArchiveIterator(warc_file.raw))
    for record in iterobject:
        if record.rec_headers.get_header(header) == value:
            return record


def get_WET_text(record):
    if record:
        # Reads text, decodes it using utf-8, makes it lowercase
        return record.content_stream().read().decode('utf-8').lower()
    else:
        return None


length = "large"
file = pd.read_csv("matches_large3.csv", sep=',', header=None, names=['word', 'text'])
dataset = file.drop_duplicates().reset_index(drop=True)


# Replace non-alphanum signs with whitespace and split into list of words
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


found_sents = []

for index, row in dataset.iterrows():
    # wet_file = get_file(row['url'])
    # wet_record = get_record_with_header(
    #     wet_file,
    #     header='WARC-Identified-Content-Language',
    #     value="eng"
    # )
    # full_text = get_WET_text(wet_record).replace("\n", ".")

    # Split into sentences using fullstops
    sentences_list = row.text.split('.')
    for sentence in sentences_list:
        # Check if there is an exact match of the word in the sentence
        if re.search(r'\b' + row['word'] + r'\b', sentence):
            found_sents.append(text_to_word_list(sentence))

phrases = Phrases(found_sents, min_count=1, progress_per=50000)
bigram = Phraser(phrases)
sentences = bigram[found_sents]
print(sentences[1])

w2v_model = Word2Vec(min_count=3,
                     window=4,
                     vector_size=300,
                     sample=1e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=multiprocessing.cpu_count()-1)

start = time()
w2v_model.build_vocab(sentences, progress_per=50000)
print('Time to build vocab: {} mins'.format(round((time() - start) / 60, 2)))

start = time()
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - start) / 60, 2)))
w2v_model.init_sims(replace=True)
w2v_model.save("Models/"+length+".word2vec.model")

file_export = pd.DataFrame(columns=['title'])
for index, sent in enumerate(found_sents):
    file_export.at[index+1, 'title'] = sent
file_export['old_title'] = file_export.title
file_export.old_title = file_export.old_title.str.join(' ')
file_export.title = file_export.title.apply(lambda x: ' '.join(bigram[x]))

file_export[['title']].to_csv('Data/'+length+'_cleaned_dataset.csv', index=False)
