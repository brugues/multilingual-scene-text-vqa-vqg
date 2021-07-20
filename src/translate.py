from utils.translator import GoogleCloudTranslate
from config.config_translate import ConfigTranslate


if __name__ == '__main__':
    config = ConfigTranslate().get_config()
    translator = GoogleCloudTranslate(config)

    translator.translate_dataset()
