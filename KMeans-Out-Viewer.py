import random
import pandas as pd
from datetime import datetime
seed = random.randint(0, 1000000)

length="low"
version="2"
dataset = pd.read_csv("Output/KMeans"+length+version+"_predictions.csv")

num_sents = 10
# Get sents with negative sentiment
neg_sents = dataset[dataset.prediction == 0]
# Random sample num_sents
sents = neg_sents.sample(n=num_sents, random_state=seed).sentence
output_chars = 100
for index, sent in enumerate(sents):
    print(str(index) + ": " + sent[:output_chars])
