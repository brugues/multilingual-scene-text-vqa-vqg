import os
import numpy as np
from bpemb import BPEmb
from sklearn.metrics.pairwise import cosine_similarity


if __name__ in '__main__':
    model_folder = '../models/bin/bpemb'
    ca_emb_path = os.path.join(model_folder, 'ca.wiki.bpe.vs200000.d300.w2v.bin')
    ca_model_path = os.path.join(model_folder, 'ca.wiki.bpe.vs200000.model')
    es_emb_path = os.path.join(model_folder, 'es.wiki.bpe.vs200000.d300.w2v.bin')
    es_model_path = os.path.join(model_folder, 'es.wiki.bpe.vs200000.model')
    en_emb_path = os.path.join(model_folder, 'en.wiki.bpe.vs200000.d300.w2v.bin')
    en_model_path = os.path.join(model_folder, 'en.wiki.bpe.vs200000.model')
    multi_emb_path = os.path.join(model_folder, 'multi.wiki.bpe.vs1000000.d300.w2v.bin')
    multi_model_path = os.path.join(model_folder, 'multi.wiki.bpe.vs1000000.model')

    ca_model = BPEmb(emb_file=ca_emb_path, model_file=ca_model_path, dim=300)
    es_model = BPEmb(emb_file=es_emb_path, model_file=es_model_path, dim=300)
    en_model = BPEmb(emb_file=en_emb_path, model_file=en_model_path, dim=300)
    multi_model = BPEmb(emb_file=multi_emb_path, model_file=multi_model_path, dim=300)

    ca_emb = ca_model.embed('gat')
    es_emb = es_model.embed('gato')
    en_emb = en_model.embed('cat')
    ca_es = cosine_similarity(ca_emb, es_emb)
    ca_en = cosine_similarity(ca_emb, en_emb)
    en_es = cosine_similarity(en_emb, es_emb)
    print("Similarity CA-ES: {}".format(ca_es))
    print("Similarity CA-EN: {}".format(ca_en))
    print("Similarity EN-ES: {}".format(en_es))

    multi_ca_emb = multi_model.embed('pet')
    multi_es_emb = multi_model.embed('pedo')
    multi_en_emb = multi_model.embed('fart')
    multi_ca_es = cosine_similarity(multi_ca_emb, multi_es_emb)
    multi_ca_en = cosine_similarity(multi_ca_emb, multi_en_emb)
    multi_en_es = cosine_similarity(multi_en_emb, multi_es_emb)
    print("Similarity CA-ES (Multilanguage): {}".format(multi_ca_es))
    print("Similarity CA-EN (Multilanguage): {}".format(multi_ca_en))
    print("Similarity EN-ES (Multilanguage): {}".format(multi_en_es))


