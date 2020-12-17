""" This script downloads text from the given url, extract passages, find their and question 
    embeddings with given model and list top passages (answers) to question 
"""
import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import time
import gzip
import os
import trafilatura

# model = SentenceTransformer('/home/ubuntu/sentence-transformers/examples/training/multilingual/output/make-multilingual-base-teacher-0_BERTen-ar-bg-cs-de-el-en-es-fi-fr-hr-hu-it-nl-pl-pt-ro-ru-sr-sv-tr-2020-11-21_10-39-25/student/')
model = SentenceTransformer('distilroberta-base-msmarco-v2')
model = SentenceTransformer('/home/ubuntu/xaynetwork/sentence-transformers/examples/training/distillation/models/distilroberta-base-msmarco-v2-64dim/')
url = 'https://www.thespruceeats.com/ginger-recipes-and-cooking-tips-1807768'

# Download the passages from given url
downloaded = trafilatura.fetch_url(url)
webpage_text = trafilatura.extract(downloaded)
split_text = webpage_text.splitlines()
passages = []
for passage in split_text:
    passages.append(passage)

# Find the passage embeddings
passage_embeddings = model.encode(passages, show_progress_bar=True)


while True:
    query = input("Please enter a question: ")
    #Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, passage_embeddings, top_k=2, embedding_size = 128)
    hits = hits[0]

    #Output of top-k hits
    print("Input question:", query)
    for hit in hits[0:2]:
        print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']]))