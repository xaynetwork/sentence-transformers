""" This script downloads text from the given url, extract passages, find their and question 
    embeddings with given model and list top passages (answers) to question using cross-encoder
"""
import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import time
import gzip
import os
import trafilatura
import numpy as np

model = CrossEncoder('amberoad/bert-multilingual-passage-reranking-msmarco', max_length=350)
url = 'https://de.wikipedia.org/wiki/Deutschland'

# Download the passages from given url
downloaded = trafilatura.fetch_url(url)
webpage_text = trafilatura.extract(downloaded)
split_text = webpage_text.splitlines()
passages = []
for passage in split_text:
    passages.append(passage)

# print(passages)
while True:
    query = input("Please enter a question: ")
   
    # So we create the respective sentence combinations
    sentence_combinations = [[query, corpus_sentence] for corpus_sentence in split_text]
    
    # Compute the similarity scores for these combinations
    if model.config.num_labels > 1: #Cross-Encoder that predict more than 1 score, we use the last and apply softmax
        similarity_scores = model.predict(sentence_combinations, apply_softmax=True)[:, 1].tolist()
    else:
        similarity_scores = model.predict(sentence_combinations).tolist()
    
    # Sort the scores in decreasing order
    sim_scores_argsort = np.argsort(similarity_scores)
    # print(sim_scores_argsort)
    sim_scores_argsort_flip = np.flip(sim_scores_argsort)
    # print(sim_scores_argsort_flip)
    # Print the scores
    print("Query:", query)
    for i in range(5):
        print("{:.2f}\t{}".format(similarity_scores[sim_scores_argsort_flip[i]], passages[sim_scores_argsort_flip[i]]))
