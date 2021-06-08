import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == '__main__':
    words = ['cafe', 'historia', 'ordinador', 'pesol']
    languages = ['ca', 'es', 'en']

    data = {
        # 'aligned': {
        #     'models': ['../models/bin/fasttext/aligned_word_vectors/wiki.ca.align.vec',
        #                '../models/bin/fasttext/aligned_word_vectors/wiki.es.align.vec',
        #                '../models/bin/fasttext/aligned_word_vectors/wiki.en.align.vec'],
        #     'languages': ['ca', 'es', 'en'],
        #     'results': np.zeros((3, len(words), 300), dtype=np.float32)
        # },
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

    for type_of_vector in data:
        models = data[type_of_vector]['models']
        for i, model in enumerate(models):
            ft_model = fasttext.load_model(model)
            for j, word in enumerate(words):
                emb = ft_model.get_word_vector(word)
                data[type_of_vector]['results'][i][j][:] = emb

    print("Finshed computing vectors")

    print("Computing distances between the two type of vectors (same language)")
    for j, word in enumerate(words):
        print("\nDistance between {} and {} in word {}".format('wiki', 'cc', word))

        for i, language in enumerate(languages):
            dist = cosine_similarity(np.expand_dims(data['wiki']['results'][i][j], axis=0),
                                     np.expand_dims(data['cc']['results'][i][j], axis=0))
            print("Distance in {} (word {}) is {})".format(language, word, dist))

    print("Computing distances between words on different languages")
    for j, word in enumerate(words):
        for i, type_of_vector in enumerate(data):
            print("Starting with {} vectors".format(type_of_vector))
