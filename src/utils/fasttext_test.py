import os
import pickle
import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.keyedvectors import KeyedVectors


if __name__ == '__main__':
    aligned = False
    out_folder = '../outputs/fasttext'
    os.makedirs(out_folder, exist_ok=True)

    if aligned:
        if not os.path.isfile(os.path.join(out_folder, 'ca.pkl')):
            print("Loading Català")
            ca_word_vector = KeyedVectors.load_word2vec_format('../models/bin/fasttext/aligned_word_vectors/'
                                                               'wiki.ca.align.vec', binary=False)
            print("Loading English")
            en_word_vector = KeyedVectors.load_word2vec_format('../models/bin/fasttext/aligned_word_vectors/'
                                                               'wiki.en.align.vec', binary=False)
            print("Loading Spanish")
            es_word_vector = KeyedVectors.load_word2vec_format('../models/bin/fasttext/aligned_word_vectors/'
                                                               'wiki.es.align.vec', binary=False)

            ca = np.expand_dims(ca_word_vector.get_vector('gat'), axis=0)
            en = np.expand_dims(en_word_vector.get_vector('cat'), axis=0)
            es = np.expand_dims(es_word_vector.get_vector('gato'), axis=0)

            with open(os.path.join(out_folder, 'ca.pkl'), 'wb+') as file:
                pickle.dump(ca, file)

            with open(os.path.join(out_folder, 'en.pkl'), 'wb+') as file:
                pickle.dump(en, file)

            with open(os.path.join(out_folder, 'es.pkl'), 'wb+') as file:
                pickle.dump(es, file)

        else:
            with open(os.path.join(out_folder, 'ca.pkl'), 'wb+') as file:
                ca = pickle.load(file)

            with open(os.path.join(out_folder, 'en.pkl'), 'wb+') as file:
                en = pickle.load(file)

            with open(os.path.join(out_folder, 'es.pkl'), 'rb') as file:
                es = pickle.load(file)

        ca_en = cosine_similarity(ca, en)
        ca_es = cosine_similarity(ca, es)
        en_es = cosine_similarity(en, es)

        print("Finished word gat/cat/gato")

    else:
        words = ['cafe', 'historia', 'ordinador', 'pesol']
        languages = ['ca', 'es', 'en']

        data = {
            'wiki': {
                'models': ['../models/bin/fasttext/wiki_word_vectors/wiki.ca.bin',
                           '../models/bin/fasttext/wiki_word_vectors/wiki.es.bin',
                           '../models/bin/fasttext/wiki_word_vectors/wiki.en.bin'],
                'languages': ['ca', 'es', 'en'],
                'results': np.zeros((3, len(words), 300), dtype=np.float32)
            },
            'cc': {
                'models': ['../models/bin/fasttext/word_vectors/cc.ca.300.bin',
                           '../models/bin/fasttext/word_vectors/cc.es.300.bin',
                           '../models/bin/fasttext/word_vectors/cc.en.300.bin'],
                'languages': ['ca', 'es', 'en'],
                'results': np.zeros((3, len(words), 300), dtype=np.float32)
            }
        }

        output_folder = '../outputs/fasttext'
        os.makedirs(output_folder, exist_ok=True)

        if not os.path.isfile(os.path.join(output_folder, 'data.pkl')):
            for type_of_vector in data:
                models = data[type_of_vector]['models']
                for i, model in enumerate(models):
                    ft_model = fasttext.load_model(model)
                    for j, word in enumerate(words):
                        emb = ft_model.get_word_vector(word)
                        data[type_of_vector]['results'][i][j][:] = emb

            with open(os.path.join(output_folder, 'data.pkl'), 'wb+') as file:
                pickle.dump(data, file)
        else:
            with open(os.path.join(output_folder, 'data.pkl'), 'rb') as file:
                data = pickle.load(file)

        print("Finished computing or loading vectors")

        print("Computing distances between the two type of vectors (same language)")
        for j, word in enumerate(words):
            print("\nDistance between {} and {} in word {}".format('wiki', 'cc', word))

            for i, language in enumerate(languages):
                dist = cosine_similarity(np.expand_dims(data['wiki']['results'][i][j], axis=0),
                                         np.expand_dims(data['cc']['results'][i][j], axis=0))
                print("Distance in {} (word {}) is {})".format(language, word, dist))

        data['wiki']['results'] = np.transpose(data['wiki']['results'], (1, 0, 2))
        data['cc']['results'] = np.transpose(data['cc']['results'], (1, 0, 2))

        pairs = ['ca-en', 'ca-es', 'en-es']
        positions = {
            'ca': 0,
            'es': 1,
            'en': 2
        }
        print("\n\nComputing distances between words on different languages (Wikipedia)")
        for i, word in enumerate(words):
            for pair in pairs:
                langs = pair.split('-')
                idx_0 = positions[langs[0]]
                idx_1 = positions[langs[1]]

                dist = cosine_similarity(np.expand_dims(data['wiki']['results'][i][idx_0], axis=0),
                                         np.expand_dims(data['wiki']['results'][i][idx_1], axis=0))

                print("Distance between {} and {} in word {}: {}".format(langs[0], langs[1], word, dist))

        print("\n\nComputing distances between words on different languages (Common Crawl)")
        for i, word in enumerate(words):
            for pair in pairs:
                langs = pair.split('-')
                idx_0 = positions[langs[0]]
                idx_1 = positions[langs[1]]

                dist = cosine_similarity(np.expand_dims(data['wiki']['results'][i][idx_0], axis=0),
                                         np.expand_dims(data['wiki']['results'][i][idx_1], axis=0))

                print("Distance between {} and {} in word {}: {}".format(langs[0], langs[1], word, dist))

        print("\n\nRecovering embeddings to see which word they produce")
        if not os.path.isfile(os.path.join(out_folder, 'english_words.pkl')):
            embs = np.zeros((6, 300), dtype=np.float32)
            similar_words = []
            word = "church"
            for type_of_vector in data:
                models = data[type_of_vector]['models']
                for i, model in enumerate(models):
                    ft_model = fasttext.load_model(model)
                    embs[0] = ft_model.get_word_vector(word)
                    similar_words.append(ft_model.get_nearest_neighbors(word, 3))

            with open(os.path.join(out_folder, 'english_words.pkl'), 'wb+') as file:
                pickle.dump(embs, file)

            with open(os.path.join(out_folder, 'english_neighbours.pkl'), 'wb+') as file:
                pickle.dump(similar_words, file)
        else:
            with open(os.path.join(out_folder, 'english_words.pkl'), 'rb') as file:
                embs = pickle.load(file)

            with open(os.path.join(out_folder, 'english_neighbours.pkl'), 'rb') as file:
                similar_words = pickle.load(file)

        print("Done")
