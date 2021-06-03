import argparse


class ConfigTranslate:

    def __init__(self):
        pass

    @staticmethod
    def get_config():
        args = argparse.ArgumentParser()

        args.add_argument('--dataset', type=str, default='eval', choices=['train', 'val'], help='Dataset to translate')
        args.add_argument('--source_language', type=str, default='en', help='Language to translate from')
        args.add_argument('--dest_language', type=str, default='es', choices=['ca', 'es'], help='Language to '
                                                                                                'translate to')
        args.add_argument('--json_config_file', type=str, default='config/google_cloud_config.json',
                          help='JSON file containing Google Cloud account info')

        return args.parse_args()
