import argparse


class ConfigOCR:

    def __init__(self):
        pass

    @staticmethod
    def get_config():
        args = argparse.ArgumentParser()

        args.add_argument('--dataset', type=str, default='train', choices=['train', 'test'],
                          help='Dataset to translate')
        args.add_argument('--language', type=str, default='zh')
        args.add_argument('--json_config_file', type=str, default='config/google_cloud_config.json',
                          help='JSON file containing Google Cloud account info')

        return args.parse_args()
