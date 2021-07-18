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
    # aligned_cc_ca = "/datasets/res/cc.en-ca.vec"
    # aligned_cc_es = "/datasets/res/cc.en-es.vec"
    # aligned_wiki_ca = "/datasets/res/wiki.ca-en.vec"
    # aligned_wiki_es = "/datasets/res/wiki.en-es.vec"
    #
    # cc_ca = "../models/bin/fasttext/cc_word_vectors/cc.en.300.bin"
    # cc_ca_vec = "../models/bin/fasttext/cc_word_vectors/cc.en.300.vec"
    # cc_ca_matrix = '/datasets/res/cc.en-ca.vec-mat'
    # cc_es = "../models/bin/fasttext/cc_word_vectors/cc.es.300.bin"
    # cc_es_vec = "../models/bin/fasttext/cc_word_vectors/cc.es.300.vec"
    # cc_es_matrix = '/datasets/res/cc.en-es.vec-mat'
    #
    # wiki_ca = "/datasets/wiki_word_vectors/wiki.ca.bin"
    # wiki_ca_vec = "/datasets/wiki_word_vectors/wiki.ca.vec"
    # wiki_ca_matrix = '/datasets/res/wiki.ca-en.vec-mat'
    # wiki_es = "/datasets/wiki_word_vectors/wiki.es.bin"
    # wiki_es_vec = "/datasets/wiki_word_vectors/wiki.es.vec"
    # wiki_es_matrix = '/datasets/res/wiki.en-es.vec-mat'

    # print("One")
    # aligned_wiki_ca = KeyedVectors.load_word2vec_format(aligned_wiki_ca, binary=False)
    # print("Two")
    # wiki_ca = fasttext.load_model(wiki_ca)
    # print("Two-and-a-half")
    # wiki_ca_vec = KeyedVectors.load_word2vec_format(wiki_ca_vec, binary=False)
    # # wiki_en = "/datasets/wiki_word_vectors/wiki.en.vec"
    # # wiki_en = KeyedVectors.load_word2vec_format(wiki_en, binary=False)
    # print("Three")
    # wiki_ca_matrix = load_transform(wiki_ca_matrix)

    for emb in ['wiki', 'cc']:
        for lang in ['ca', 'es']:
            print("Doing {} {}-en".format(emb, lang))
            aligned = "/datasets/res/{}.{}-en.vec".format(emb, lang)

            if emb == 'cc':
                not_aligned_bin = "../models/bin/fasttext/{}_word_vectors/{}.{}.300.bin".format(emb, emb, lang)
                not_aligned_vec = "../models/bin/fasttext/{}_word_vectors/{}.{}.300.vec".format(emb, emb, lang)
            else:
                not_aligned_bin = "../models/bin/fasttext/{}_word_vectors/{}.{}.bin".format(emb, emb, lang)
                not_aligned_vec = "../models/bin/fasttext/{}_word_vectors/{}.{}.vec".format(emb, emb, lang)

            aligned = KeyedVectors.load_word2vec_format(aligned, binary=False)
            not_aligned_bin = fasttext.load_model(not_aligned_bin)
            not_aligned_vec = KeyedVectors.load_word2vec_format(not_aligned_vec, binary=False)
            transform = load_transform('/datasets/res/{}.{}-en.vec-mat'.format(emb, lang))

            if lang == 'ca':
                word = 'gat'
            else:
                word = 'gato'

            aligned = aligned[word]
            not_aligned_bin = not_aligned_bin.get_word_vector(word)
            not_aligned_vec = not_aligned_vec[word]
            not_aligned_bin = np.dot(not_aligned_bin, transform.T)
            not_aligned_vec = np.dot(not_aligned_vec, transform.T)

            print("Bin-Vec similarity: {}".format(cosine_similarity(np.expand_dims(not_aligned_vec, axis=0),
                                                                    np.expand_dims(not_aligned_bin, axis=0))))
            print("Bin-Aligned similarity: {}".format(cosine_similarity(np.expand_dims(aligned, axis=0),
                                                                        np.expand_dims(not_aligned_bin, axis=0))))
            print("Vec-Aligned similarity: {}".format(cosine_similarity(np.expand_dims(aligned, axis=0),
                                                                        np.expand_dims(not_aligned_vec, axis=0))))

#     # en = wiki_en['cat']
#     ca_al = aligned_wiki_ca['gat']
#     # print(cosine_similarity(np.expand_dims(en, axis=0), np.expand_dims(ca_al, axis=0)))
#
#     ca = wiki_ca_vec['gat']
#     ca_bin = wiki_ca.get_word_vector('gat')
#
#     print(cosine_similarity(np.expand_dims(ca_bin, axis=0), np.expand_dims(ca, axis=0)))
#     print(cosine_similarity(np.expand_dims(np.dot(ca, wiki_ca_matrix.T), axis=0), np.expand_dims(ca_al, axis=0)))
# )


# CATALA
# print("Starting with catalan")
# words = ['la']
# aligned_wiki_ca = KeyedVectors.load_word2vec_format(aligned_wiki_ca, binary=False)
# wiki_ca = fasttext.load_model(wiki_ca)
# # wiki_ca_vec = KeyedVectors.load_word2vec_format(wiki_ca_vec)
# wiki_ca_matrix = load_transform(wiki_ca_matrix)
#
# for word in words:
#     # vec_emb = wiki_ca_vec[word]
#     bin_emb = wiki_ca.get_word_vector(word)
#     aligned_emb = aligned_wiki_ca['la']
#     transformed = np.dot(bin_emb, wiki_ca_matrix)
#     # transformed /= np.linalg.norm(transformed)[np.newaxis] + 1e-8
#     print("A")
#
# del wiki_ca
# del aligned_wiki_ca
#
# aligned_cc_ca = KeyedVectors.load_word2vec_format(aligned_cc_ca, binary=False)
# cc_ca = fasttext.load_model(cc_ca)
# # cc_ca_vec = KeyedVectors.load_word2vec_format(cc_ca_vec)
# cc_ca_matrix = load_transform(cc_ca_matrix)
# cc_ca = fasttext.load_model(cc_ca)
#
# print("Analyzing Catalan")
# for word in words:
#     aligned_emb = aligned_cc_ca.get_vector(word)
#     cc_emb = cc_ca.get_word_vector(word)
#     cc_emb = np.dot(cc_emb, cc_ca_matrix)
#     dist = cosine_similarity(np.expand_dims(aligned_emb, axis=0),
#                              np.expand_dims(cc_emb, axis=0))
#     print("Similarity in word {}: {}".format(word, dist))
#
# del cc_ca
# del cc_ca_matrix
# del aligned_cc_ca

# # ESPAÑOL
# words = ['de', 'para', 'años', 'son']
# aligned_es = KeyedVectors.load_word2vec_format(aligned_es, binary=False)
# cc_es = fasttext.load_model(cc_es)
#
# for word in words:
#     aligned_emb = aligned_es.get_vector(word)
#     cc_emb = cc_es.get_word_vector(word)
#
#
# del cc_es
# wiki_es = fasttext.load_model(wiki_es)
#
# for word in words:
#     pass


if __name__ == "__main__":
    main()
