import io
import fasttext
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity


def load_transform(fname, d1=300, d2=300):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    R = np.zeros([d1, d2])
    for i, line in enumerate(fin):
        tokens = line.split(' ')
        R[i, :] = np.array(tokens[0:d2], dtype=float)
    return R


def main():
    aligned_ca = "../models/bin/fasttext/aligned_word_vectors/wiki.ca.align.vec"
    aligned_es = "../models/bin/fasttext/aligned_word_vectors/wiki.es.align.vec"

    cc_ca = "../models/bin/fasttext/cc_word_vectors/cc.ca.300.bin"
    cc_ca_matrix = '../fastText/alignment/res/cc.en-ca.vec-mat'
    cc_es = "../models/bin/fasttext/cc_word_vectors/cc.es.300.bin"
    cc_es_matrix = '../fastText/alignment/res/cc.en-es.vec-mat'

    wiki_ca = "../models/bin/fasttext/wiki_word_vectors/wiki.ca.bin"
    wiki_ca_matrix = '../fastText/alignment/res/wiki.en-ca.vec-mat'
    wiki_es = "../models/bin/fasttext/wiki_word_vectors/wiki.es.bin"
    wiki_es_matrix = '../fastText/alignment/res/wiki.en-es.vec-mat'

    # CATALA
    words = ['la', 'amb', 'seva', 'també']
    aligned_ca = KeyedVectors.load_word2vec_format(aligned_ca, binary=False)
    cc_ca = fasttext.load_model(cc_ca)
    cc_ca_matrix = load_transform(cc_ca_matrix)

    print("Analyzing Catalan")
    for word in words:
        aligned_emb = aligned_ca.get_vector(word)
        cc_emb = cc_ca.get_word_vector(word)
        cc_emb = np.dot(cc_emb, cc_ca_matrix)
        dist = cosine_similarity(np.expand_dims(aligned_emb, axis=0),
                                 np.expand_dims(cc_emb, axis=0))
        print("Similarity in word {}: {}".format(word, dist))

    del cc_ca

    wiki_ca = fasttext.load_model(wiki_ca)
    wiki_ca_matrix = load_transform(wiki_ca_matrix)

    for word in words:
        pass

    del wiki_ca
    del aligned_ca

    # ESPAÑOL
    words = ['de', 'para', 'años', 'son']
    aligned_es = KeyedVectors.load_word2vec_format(aligned_es, binary=False)
    cc_es = fasttext.load_model(cc_es)

    for word in words:
        aligned_emb = aligned_es.get_vector(word)
        cc_emb = cc_es.get_word_vector(word)


    del cc_es
    wiki_es = fasttext.load_model(wiki_es)

    for word in words:
        pass


if __name__ == "__main__":
    main()
