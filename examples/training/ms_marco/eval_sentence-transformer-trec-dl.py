"""
This file evaluates CrossEncoder on the TREC 2019 Deep Learning (DL) Track: https://arxiv.org/abs/2003.07820

TREC 2019 DL is based on the corpus of MS Marco. MS Marco provides a sparse annotation, i.e., usually only a single
passage is marked as relevant for a given query. Many other highly relevant passages are not annotated and hence are treated
as an error if a model ranks those high.

TREC DL instead annotated up to 200 passages per query for their relevance to a given query. It is better suited to estimate
the model performance for the task of reranking in Information Retrieval.

Run:
python eval_cross-encoder-trec-dl.py cross-encoder-model-name

"""
import gzip
from collections import defaultdict
import logging
import tqdm
import numpy as np
import sys
import pytrec_eval
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import os

data_folder = 'trec2019-data'
os.makedirs(data_folder, exist_ok=True)

#Read test queries
queries = {}
queries_filepath = os.path.join(data_folder, 'msmarco-test2019-queries.tsv.gz')
if not os.path.exists(queries_filepath):
    logging.info("Download "+os.path.basename(queries_filepath))
    util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz', queries_filepath)

with gzip.open(queries_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query

#Read which passages are relevant
relevant_docs = defaultdict(lambda: defaultdict(int))
qrels_filepath = os.path.join(data_folder, '2019qrels-pass.txt')

if not os.path.exists(qrels_filepath):
    logging.info("Download "+os.path.basename(qrels_filepath))
    util.http_get('https://trec.nist.gov/data/deep/2019qrels-pass.txt', qrels_filepath)


with open(qrels_filepath) as fIn:
    for line in fIn:
        qid, _, pid, score = line.strip().split()
        score = int(score)
        if score > 0:
            relevant_docs[qid][pid] = score

# Only use queries that have at least one relevant passage
relevant_qid = []
for qid in queries:
    if len(relevant_docs[qid]) > 0:
        relevant_qid.append(qid)


# Read the top 1000 passages that are supposed to be re-ranked
passage_filepath = os.path.join(data_folder, 'msmarco-passagetest2019-top1000.tsv.gz')

if not os.path.exists(passage_filepath):
    logging.info("Download "+os.path.basename(passage_filepath))
    util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz', passage_filepath)



passage_cand = {}
with gzip.open(passage_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        qid, pid, query, passage = line.strip().split("\t")
        if qid not in passage_cand:
            passage_cand[qid] = []

        passage_cand[qid].append([pid, passage])

logging.info("Queries: {}".format(len(queries)))


run = {}
run1 = {}
model = SentenceTransformer('/home/ubuntu/sentence-transformers/examples/training/multilingual/output/make-multilingual-base-teacher-0_BERTen-ar-bg-cs-de-el-en-es-fi-fr-hr-hu-it-nl-pl-pt-ro-ru-sr-sv-tr-2020-11-21_10-39-25/student/')
# model = SentenceTransformer('/home/ubuntu/sentence-transformers/examples/training/multilingual/output/make-multilingual-base-teacher-0_BERTen-ar-bg-cs-de-el-en-es-fi-fr-hr-hu-it-nl-pl-pt-ro-ru-sr-sv-tr-2020-12-17_10-29-40/student')
for qid in tqdm.tqdm(relevant_qid):
    query = queries[qid]

    cand = passage_cand[qid]
    pids = [c[0] for c in cand]
    corpus_sentences = [c[1] for c in cand]
    
    question_embedding = model.encode(query, convert_to_tensor=True)
    corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=False)
    hits = util.calculate_score(question_embedding, corpus_embeddings, embedding_size = 128)
    hits = hits[0]  # Get the hits for the first query

    scores_sparse = {}
    euclidean_scores = {}
    for idx, pid in enumerate(pids):
        scores_sparse[pid] = hits[idx]['score']
        euclidean_scores[pid] = hits[idx]['Euclidean_distance']
    
    run[qid] = {}
    run1[qid] = {}
    for pid in scores_sparse:
        run[qid][pid] = float(scores_sparse[pid])
        run1[qid][pid] = float(euclidean_scores[pid])


evaluator = pytrec_eval.RelevanceEvaluator(relevant_docs, {'ndcg_cut.10'})
scores = evaluator.evaluate(run)

print("Queries:", len(relevant_qid))
print("NDCG@10: {:.2f}".format(np.mean([ele["ndcg_cut_10"] for ele in scores.values()])*100))

scores1 = evaluator.evaluate(run1)

print("Queries:", len(relevant_qid))
print("NDCG@10: {:.2f}".format(np.mean([ele["ndcg_cut_10"] for ele in scores1.values()])*100))


