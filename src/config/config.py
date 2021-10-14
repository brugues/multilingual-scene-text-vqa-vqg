import argparse


class Config:

    def __init(self):
        pass

    @staticmethod
    def get_config():
        args = argparse.ArgumentParser()

        # -------------------------------- NETWORK --------------------------------
        args.add_argument('--rnn_size', type=int, default=256, help='Units of the LSTM')
        args.add_argument('--rnn_layer', type=int, default=2, help='Number of LSTM layers')

        args.add_argument('--img_size', type=int, default=608, help='Yolo input image size (img_size, img_size)')
        args.add_argument('--img_feature_shape', type=tuple, default=(38, 38, 512), help='YOLO output')
        args.add_argument('--txt_feature_shape', type=tuple, default=(38, 38, 300), help='fasttext grid embedding')
        args.add_argument('--dim_image', type=int, default=512, help='Size of visual features. Last element of img '
                                                                     'feature size')
        args.add_argument('--dim_txt', type=int, default=300, help='Size of individual text features')
        args.add_argument('--dim_hidden', type=int, default=1024, help='Size of the common embedding vector')
        args.add_argument('--dim_attention', type=int, default=512, help='Size of the attention embedding')
        args.add_argument('--text_embedding_dim', type=int, default=300, help='Size of textual features, fasttext'
                                                                              'dimension')
        args.add_argument('--max_len', type=int, default=25, help='Question maximum length')
        args.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
        args.add_argument('--num_grids', type=int, default=38, help='Num grids per dimension on the image')

        # ------------------------------  DATALOADER  -----------------------------
        args.add_argument('--dataset', type=str, default='stvqa', choices=['stvqa', 'estvqa'], help='Dataset to use')
        args.add_argument('--combined_datasets', dest='combined_datasets', action='store_true')
        args.set_defaults(combined_datasets=False)
        args.add_argument('--no_shuffle', dest='shuffle', help='Shuffle data')
        args.set_defaults(shuffle=True)
        args.add_argument('--gt_file', type=str, default='data/ST-VQA/annotations/stvqa_train.json',
                          help='GT file path')
        args.add_argument('--gt_eval_file', type=str, default='data/ST-VQA/annotations/stvqa_eval.json', help='GT eval file path')
        args.add_argument('--language', type=str, default='en', help='Language of the embeddings to use',
                          choices=['ca', 'en', 'es', 'zh', 'en-ca', 'en-es', 'en-zh'])
        args.add_argument('--embedding_type', type=str, default='fasttext', choices=['fasttext', 'bpemb', 'smith'],
                          help='What type of embeddings to use')
        args.add_argument('--fasttext_subtype', type=str, default='wiki-news', help='Subtype of fasttext embeddings',
                          choices=['wiki', 'cc', 'wiki-news'])
        args.add_argument('--fasttext_aligned', dest='fasttext_aligned', action='store_true')
        args.set_defaults(fasttext_aligned=False)
        args.add_argument('--fasttext_aligned_pair', default='ca-en', help='Pair of languages aligned (source-target)')
        args.add_argument('--bpemb_subtype', type=str, default='wiki', help='Subtype of bpemb embeddings',
                          choices=['wiki', 'multi'])
        args.add_argument('--txt_embeddings_path', type=str, default='models/bin',
                          help='FastText files')

        # -------------------------------  TRAINING  ------------------------------
        args.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
        args.add_argument('--decay_steps', type=int, default=50, help='Decay LR every X steps')
        args.add_argument('--staircase', type=bool, default=True, help='Exponential decay with staircase')
        args.add_argument('--apply_decay', type=bool, default=True, help='Apply decay rate to learning rate')
        args.add_argument('--decay_factor', type=float, default=0.99997592083, help='Learning rate decay factor')
        args.add_argument('--batch_size', type=int, default=32, help='batch size')
        args.add_argument('--n_epochs', type=int, default=100, help='Number of epochs for which to train the net for')
        args.add_argument('--models_path', type=str, default='outputs/models', help='models directory')
        args.add_argument('--logging_path', type=str, default='outputs/logs', help='path to save logs')
        args.add_argument('--checkpoint_period', type=int, default=200, help='save checkpoint every X steps')
        args.add_argument('--logging_period', type=int, default=50, help='log to tensorboard every X steps')
        args.add_argument('--load_checkpoint', type=bool, default=False, help='Continue last training by loading '
                                                                              'checkpoint')
        args.add_argument('--output_folder', type=str, default=None, help='Name of the output folder inside models and '
                                                                          'logs')

        # ------------------------------  EVALUATION  ----------------------------
        args.add_argument('--model_to_evaluate', type=str, default='./outputs/models/config2')
        args.add_argument('--no_server_evaluation', dest='server_evaluation', action='store_false')
        args.set_defaults(server_evaluation=True)

        # --------------------------------  PATHS  -------------------------------
        args.add_argument('--image_path', type=str, default='data/ST-VQA', help='Image paths')
        args.add_argument('--yolo_file', type=str, default='models/bin/yolov4_tf231.h5', help='Yolo weight file')

        return args.parse_args()
