"""
This basic example computes the embeddings of sentence transformer model.
"""

from sentence_transformers import SentenceTransformer, LoggingHandler

model = SentenceTransformer('/home/ubuntu/sentence-transformers/examples/training/multilingual/output/make-multilingual-base-teacher-0_BERTen-ar-bg-cs-de-el-en-es-fi-fr-hr-hu-it-nl-pl-pt-ro-ru-sr-sv-tr-2020-11-21_10-39-25/student') # S-mBERT 1/128 

sentence_embeddings= model.encode("A test example")