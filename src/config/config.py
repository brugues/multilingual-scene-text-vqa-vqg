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
        args.add_argument('--dim_hidden', type=int, default=1024, help='Size of the common embedding vector')
        args.add_argument('--dim_attention', type=int, default=512, help='Size of the attention embedding')
        args.add_argument('--text_embedding_dim', type=int, default=300, help='Size of textual features, fasttext'
                                                                              'dimension')
        args.add_argument('--max_len', type=int, default=25, help='Question maximum length')
        args.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')

        # ------------------------------  DATALOADER  -----------------------------
        args.add_argument('--shuffle', type=bool, default=True, help='Shuffle data')
        args.add_argument('--gt_file', type=str, default='data/stvqa_train.json', help='GT file path')

        # -------------------------------  TRAINING  ------------------------------
        args.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
        args.add_argument('--decay_steps', type=int, default=50, help='Decay LR every X steps')
        args.add_argument('--decay_factor', type=float, default=0.99997592083, help='Learning rate decay factor')
        args.add_argument('--batch_size', type=int, default=32, help='batch size')
        args.add_argument('--n_epochs', type=int, default=100, help='Number of epochs for which to train the net for')
        args.add_argument('--models_path', type=str, default='outputs/models', help='models directory')
        args.add_argument('--logging_path', type=str, default='outputs/logs', help='path to save logs')
        args.add_argument('--checkpoint_period', type=int, default=200, help='save checkpoint every X steps')
        args.add_argument('--logging_period', type=int, default=50, help='log to tensorboard every X steps')

        # --------------------------------  PATHS  -------------------------------
        args.add_argument('--gt_file_train', type=str, default='data/stvqa_train.json', help='Ground Truth data files')
        args.add_argument('--image_path', type=str, default='data/ST-VQA', help='Image paths')
        args.add_argument('--yolo_file', type=str, default='models/bin/yolov4.h5', help='Yolo weight file')
        args.add_argument('--fasttext_file', type=str, default='models/bin/wiki-news-300d-1M-subword.bin',
                          help='FastText files')

        return args.parse_args()
