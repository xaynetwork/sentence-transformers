"""
This basic example compares the sentence embeddings generated by
two pre-trained models for a given list of sentences.
"""

from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

def cosine_similarity(x, y):
    assert(x.ndim==1)   
    assert(y.ndim==1)
    assert(x.shape==y.shape)
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def mse_similarity(x, y):
    assert(x.ndim==1)   
    assert(y.ndim==1)
    assert(x.shape==y.shape)
    return np.sqrt(np.mean(np.square(x-y)))

# Load pre-trained Sentence Transformer Model 
model_1 = SentenceTransformer('/home/ubuntu/sentence-transformers/examples/training/multilingual/output/make-multilingual-base-teacher-0_BERTen-ar-bg-cs-de-el-en-es-fi-fr-hr-hu-it-nl-pl-pt-ro-ru-sr-sv-tr-2020-11-21_10-39-25/student/')
model_2 = SentenceTransformer('/home/amitchaulwar/sentence-transformers/examples/training/multilingual/output/make-multilingual-en-ar-bg-cs-da-de-el-en-es-fi-fr-hr-hu-it-nl-pl-pt-ro-ru-sr-sv-tr-2021-01-12_21-40-15/')

# Embed a list of sentences/snippets
sentences = ['The official Liverpool FC website. The only place to visit for all your LFC news, videos, history and match information. Full stats on LFC players, club products, official partners and lots more.',
             'Liverpool [ ˈlɪvəpuːl] ist eine Stadt mit rund 495.000 Einwohnern und ein Metropolitan Borough im Nordwesten Englands. Liverpool liegt an der Mündung des Flusses Mersey in die Irische See. Die Agglomeration Liverpool Urban Area beherbergt rund 860.000 Einwohner. Liverpool hat den zweitgrößten Exporthafen Großbritanniens.',
             'Alle Informationen zum FC Liverpool auf einem Blick! Hier gelangen Sie zu den aktuellen News, Spielplan, Kader und Liveticker! Hier zur Infoseite vom LFC!',
             'FC Liverpool - Die Vereinsinfos, alle Daten, Statistiken und News - kicker',
             'Im Oktober 2015 einigten sich der FC Liverpool und Jürgen Klopp auf einen Dreijahresvertrag als Coach. Zuvor wurde der ehemalige BVB -Trainer wochenlang von den Liverpool-Fans gefordert (“Klopp for...',
             'Der FC Liverpool (offiziell: Liverpool Football Club) – auch bekannt als The Reds (englisch für Die Roten) – ist ein 1892 gegründeter Fußballverein aus Liverpool.',
             'Im Nordwesten Englands liegt Liverpool, eine der bedeutendsten Hafenstädte Großbritanniens. Die Stadt wurde 1190 gegründet und wuchs im 17. Jahrhundert zu einer florierenden Handelsstadt.',
             'Der FC Liverpool (offiziell: Liverpool Football Club) – auch bekannt als The Reds (englisch für Die Roten) – ist ein 1892 gegründeter Fußballverein aus Liverpool.',
             'Liverpools Reichtum ist historisch mit der Schifffahrt verbunden. Jedoch verblassen Import- und Exportwaren wie Zucker, Gewürze und Tabak neben Liverpools wichtigstem Exportschlager: den Beatles. Durchleben Sie im Beatles Story Experience erneut die Hysterie der damaligen Zeit und besuchen Sie das Elternhaus von Paul McCartney.', 
             'Liverpools Reichtum ist historisch mit der Schifffahrt verbunden. Jedoch verblassen Import- und Exportwaren wie Zucker, Gewürze und Tabak neben Liverpools wichtigstem Exportschlager: den Beatles. Durchleben Sie im Beatles Story Experience erneut die Hysterie der damaligen Zeit und besuchen Sie das Elternhaus von Paul McCartney.'
             ]

sentence_embeddings_1 = model_1.encode(sentences)
sentence_embeddings_2 = model_2.encode(sentences)


# The result is a list of sentence embeddings as numpy arrays
for sentence, embedding_1, embedding_2 in zip(sentences, sentence_embeddings_1, sentence_embeddings_2):
    print("Sentence:", sentence)
    print("Embedding error", mse_similarity(embedding_1, embedding_2))
    print("cosine similarity", cosine_similarity(embedding_1, embedding_2))
    print("")



